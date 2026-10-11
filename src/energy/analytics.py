"""Deterministic analytics: pricing, baseline, bill bridge, idle load, leaks, savings.

Every number an answer may quote is produced here. Nothing in this module calls an LLM.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from .pipeline import working_days

NIGHT_HOURS = (0, 5)
CAUSES = ["tariff", "volume", "product_mix", "weather", "waste"]


def price_series(index: pd.DatetimeIndex, tariff: pd.DataFrame, as_of: pd.Timestamp | None = None) -> np.ndarray:
    """Price per kWh for every interval. With as_of, the schedule in force on that date is
    applied to all intervals (used to price current energy at reference-quarter prices)."""
    hours = np.asarray(index.hour)
    out = np.full(len(index), np.nan)
    for r in tariff.itertuples():
        lo, hi = int(r.start_hour), int(r.end_hour)
        in_period = (hours >= lo) & (hours < hi) if lo < hi else (hours >= lo) | (hours < hi)
        if as_of is None:
            mask = in_period & np.asarray((index >= r.valid_from) & (index < r.valid_to))
        elif r.valid_from <= as_of < r.valid_to:
            mask = in_period
        else:
            continue
        out[mask] = r.price_per_kwh
    return out


def quarter_mask(index: pd.DatetimeIndex, quarter: str) -> np.ndarray:
    return np.asarray(index.to_period("Q") == pd.Period(quarter.replace("-", ""), freq="Q"))


def daily_table(meters: pd.DataFrame, frames: dict, reference: str) -> pd.DataFrame:
    """One row per day: energy, cost at actual and at reference prices, and the drivers."""
    tariff = frames["tariff"]
    ref_start = pd.Period(reference.replace("-", ""), freq="Q").start_time
    main = meters["main"]
    df = pd.DataFrame({"kwh": main,
                       "cost": main * price_series(meters.index, tariff),
                       "cost_ref": main * price_series(meters.index, tariff, as_of=ref_start)})
    daily = df.groupby(meters.index.normalize()).sum()
    units = frames["production"].pivot_table(index="date", columns="product", values="units", aggfunc="sum")
    units.columns = [f"units_{c}" for c in units.columns]
    daily = daily.join(units).join(frames["weather"].set_index("date")[["cooling_degree_days", "heating_degree_days"]])
    unit_cols = list(units.columns)
    daily[unit_cols] = daily[unit_cols].fillna(0.0)
    daily = daily.rename(columns={"cooling_degree_days": "cdd", "heating_degree_days": "hdd"}).fillna({"cdd": 0.0, "hdd": 0.0})
    work = working_days(frames["production"]).reindex(daily.index).fillna(False).astype(bool)
    daily[unit_cols] = daily[unit_cols].where(work, 0.0)  # a holiday logged as a shift produces nothing
    daily["work"], daily["idle"] = work.astype(float), (~work).astype(float)
    daily.attrs["unit_cols"] = unit_cols
    return daily


def _features(daily: pd.DataFrame, unit_cols: list[str]) -> list[str]:
    cols = ["work", "idle"] + unit_cols
    return cols + [c for c in ("cdd", "hdd") if daily[c].std() > 1e-9]


def fit_baseline(train: pd.DataFrame, unit_cols: list[str], target: str = "cost_ref",
                 model: str = "ols", alpha: float = 1.0) -> dict:
    """Linear baseline with no global intercept: working-day and idle-day base load,
    energy per unit of each product, and energy per degree-day."""
    cols = _features(train, unit_cols)
    X, y = train[cols].to_numpy(float), train[target].to_numpy(float)
    if model == "ridge":  # shrink only the slope terms, on standardised features
        scale = np.where(np.isin(cols, ["work", "idle"]), 1.0, X.std(axis=0) + 1e-12)
        Xs = X / scale
        penalty = np.diag([0.0 if c in ("work", "idle") else alpha for c in cols])
        beta = np.linalg.solve(Xs.T @ Xs + penalty, Xs.T @ y) / scale
    else:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return {"cols": cols, "beta": dict(zip(cols, beta.tolist())), "r2": r2, "resid_std": float(resid.std()), "model": model}


def _state(daily: pd.DataFrame, unit_cols: list[str]) -> dict:
    u = daily[unit_cols].sum().to_numpy(float)
    total = float(u.sum())
    return {"calendar": (float(daily["work"].sum()), float(daily["idle"].sum())), "volume": total,
            "mix": u / total if total else u, "weather": (float(daily["cdd"].sum()), float(daily["hdd"].sum()))}


def _predict(fit: dict, unit_cols: list[str], s: dict) -> float:
    b = fit["beta"]
    k = np.array([b[c] for c in unit_cols])
    return (b["work"] * s["calendar"][0] + b["idle"] * s["calendar"][1] + s["volume"] * float(k @ s["mix"])
            + b.get("cdd", 0.0) * s["weather"][0] + b.get("hdd", 0.0) * s["weather"][1])


def _shapley(fit: dict, unit_cols: list[str], s0: dict, s1: dict) -> dict:
    """Average each driver's effect over every ordering, so the split does not depend on order."""
    names = list(s0)
    out = dict.fromkeys(names, 0.0)
    orders = list(permutations(names))
    for order in orders:
        s = dict(s0)
        prev = _predict(fit, unit_cols, s)
        for n in order:
            s[n] = s1[n]
            now = _predict(fit, unit_cols, s)
            out[n] += (now - prev) / len(orders)
            prev = now
    return out


def _bridge_amounts(ref: pd.DataFrame, cur: pd.DataFrame, unit_cols: list[str], model: str, train: pd.DataFrame) -> dict:
    fit = fit_baseline(train, unit_cols, model=model)
    drivers = _shapley(fit, unit_cols, _state(ref, unit_cols), _state(cur, unit_cols))
    c0, c1, c1_ref = float(ref["cost"].sum()), float(cur["cost"].sum()), float(cur["cost_ref"].sum())
    amounts = {"tariff": c1 - c1_ref,
               "volume": drivers["volume"] + drivers["calendar"],  # more working days is more activity
               "product_mix": drivers["mix"], "weather": drivers["weather"]}
    amounts["waste"] = (c1 - c0) - sum(amounts.values())  # what the known causes do not explain
    return {"amounts": amounts, "fit": fit, "c0": c0, "c1": c1}


def decompose(daily: pd.DataFrame, period: str, compare_to: str, model: str = "ols",
              n_boot: int = 200, seed: int = 0) -> dict:
    """Split the bill change between two quarters into five causes that add up to the total."""
    unit_cols = daily.attrs["unit_cols"]
    ref = daily[quarter_mask(daily.index, compare_to)]
    cur = daily[quarter_mask(daily.index, period)]
    if len(ref) < 30 or len(cur) < 30:
        return {"status": "insufficient_data", "detail": f"{len(ref)} reference days, {len(cur)} current days"}
    point = _bridge_amounts(ref, cur, unit_cols, model, ref)
    c0 = point["c0"]
    rng = np.random.default_rng(seed)
    boots = {c: [] for c in CAUSES}
    for _ in range(n_boot):  # refit the baseline on resampled reference days
        sample = ref.iloc[rng.integers(0, len(ref), len(ref))]
        b = _bridge_amounts(ref, cur, unit_cols, model, sample)["amounts"]
        for c in CAUSES:
            boots[c].append(100 * b[c] / c0)
    bridge = []
    for c in CAUSES:
        pts = 100 * point["amounts"][c] / c0
        lo, hi = np.percentile(boots[c], [5, 95])
        bridge.append({"cause": c, "pct_points": round(pts, 2), "range": [round(min(lo, pts), 2), round(max(hi, pts), 2)],
                       "amount": round(point["amounts"][c], 2)})
    total = 100 * (point["c1"] - c0) / c0
    # displayed values must add up too: put the rounding remainder on the largest cause
    gap = round(round(total, 2) - sum(b["pct_points"] for b in bridge), 2)
    if gap:
        big = max(bridge, key=lambda b: abs(b["pct_points"]))
        big["pct_points"] = round(big["pct_points"] + gap, 2)
        big["range"] = [min(big["range"][0], big["pct_points"]), max(big["range"][1], big["pct_points"])]
    fit = point["fit"]
    return {"status": "ok", "period": period, "compare_to": compare_to,
            "bill_reference": round(c0, 2), "bill_current": round(point["c1"], 2),
            "total_change_pct": round(total, 2), "bridge": bridge,
            "baseline": {"model": model, "r2": round(fit["r2"], 4), "coefficients": {k: round(v, 4) for k, v in fit["beta"].items()},
                         "target": "daily cost at reference-quarter prices", "trained_on": compare_to},
            "assumptions": ["reference-quarter tariff schedule is the 'old' price for every interval",
                            "calendar effect (number of working days) is counted in volume",
                            "ranges are the 5th to 95th percentile of 200 bootstrap refits of the baseline"]}


def asset_breakdown(meters: pd.DataFrame, period: str) -> list[dict]:
    q = meters[quarter_mask(meters.index, period)]
    main = float(q["main"].sum())
    subs = q.drop(columns="main").sum().sort_values(ascending=False)
    rows = [{"asset": m, "kwh": round(float(v), 1), "share_of_main": round(float(v) / main, 4), "basis": "sub-meter"}
            for m, v in subs.items()]
    rows.append({"asset": "unmetered", "kwh": round(main - float(subs.sum()), 1),
                 "share_of_main": round(1 - float(subs.sum()) / main, 4), "basis": "estimated remainder"})
    return rows


def idle_load(meters: pd.DataFrame, frames: dict, period: str) -> list[dict]:
    """Energy used on days with no production, by asset."""
    q = meters[quarter_mask(meters.index, period)]
    work = working_days(frames["production"]).reindex(q.index.normalize()).fillna(False).to_numpy()
    idle = q[~work]
    days = max(idle.index.normalize().nunique(), 1)
    return [{"asset": m, "idle_kwh": round(float(v), 1), "idle_kwh_per_day": round(float(v) / days, 1), "idle_days": int(days)}
            for m, v in idle.drop(columns="main").sum().sort_values(ascending=False).items()]


def detect_leaks(meters: pd.DataFrame, frames: dict, period: str, compare_to: str,
                 min_rise: float = 0.10, z: float = 4.0, run: int = 5) -> list[dict]:
    """Flag a sub-meter whose night-time load steps up and stays up for at least `run` days.
    Night load is compared within the same day type (working or idle)."""
    night = meters[(meters.index.hour >= NIGHT_HOURS[0]) & (meters.index.hour < NIGHT_HOURS[1])].drop(columns="main")
    per_day = night.groupby(night.index.normalize()).mean() * 4.0  # kW
    work = working_days(frames["production"]).reindex(per_day.index).fillna(False)
    ref_days, cur_days = quarter_mask(per_day.index, compare_to), quarter_mask(per_day.index, period)
    findings = []
    for m in per_day.columns:
        excess = pd.Series(np.nan, index=per_day.index)
        zscore = pd.Series(0.0, index=per_day.index)
        base_kw = float(per_day.loc[ref_days, m].mean())
        for day_type in (True, False):
            sel = (work == day_type).to_numpy()
            ref = per_day.loc[sel & ref_days, m]
            if len(ref) < 5:
                continue
            excess[sel] = per_day.loc[sel, m] - ref.mean()
            zscore[sel] = excess[sel] / (ref.std() + 1e-9)
        cur_excess, cur_z = excess[cur_days], zscore[cur_days]
        high = (cur_z > z) & (cur_excess > min_rise * base_kw)
        streak = high.groupby((~high).cumsum()).cumsum()
        if not (streak >= run).any():
            continue
        end_of_run = streak[streak >= run].index[0]
        start = high.loc[:end_of_run][::-1].cummin()[::-1].idxmax()
        after = cur_excess.loc[start:].dropna()
        kw, se = float(after.mean()), float(after.std() / np.sqrt(len(after)))
        findings.append({"asset": m, "pattern": "night load stepped up and stayed up", "from": start.strftime("%Y-%m-%d"),
                         "excess_kw": round(kw, 2), "excess_kw_range": [round(0.9 * (kw - 2 * se), 2), round(1.1 * (kw + 2 * se), 2)],
                         "reference_night_kw": round(base_kw, 2), "rise_pct": round(100 * kw / base_kw, 1),
                         "days_observed": int(len(after)),
                         "evidence_refs": [f"meter_15min:{m}:{start.strftime('%Y-%m-%d')}..{after.index[-1].strftime('%Y-%m-%d')}:00-05h"]})
    return findings


def estimate_savings(leaks: list[dict], meters: pd.DataFrame, frames: dict, emission_factor: float,
                     action_cost: tuple[float, float] = (20_000.0, 60_000.0)) -> list[dict]:
    """Turn each leak into an action with ranges. action_cost is an assumption: the data pack has no cost table."""
    latest = meters.index.max()
    flat_price = float(np.mean(price_series(pd.date_range(latest.normalize(), periods=96, freq="15min"), frames["tariff"])))
    actions = []
    for f in leaks:
        kwh = [round(k * 8760) for k in f["excess_kw_range"]]
        money = [round(k * flat_price) for k in kwh]
        payback = [round(12 * action_cost[0] / money[1], 1), round(12 * action_cost[1] / money[0], 1)]
        actions.append({"action": f"Find and repair the leak on {f['asset']}", "asset": f["asset"],
                        "saving_kwh_per_year": kwh, "saving_money_per_year": money,
                        "saving_co2e_t_per_year": [round(k * emission_factor / 1000, 1) for k in kwh],
                        "payback_months": payback, "ease": "easy",
                        "rationale": f"Night load on {f['asset']} rose {f['rise_pct']}% from {f['from']} and stayed up on idle days.",
                        "evidence_refs": f["evidence_refs"], "confidence": "medium", "requires_approval": True,
                        "assumptions": [f"price {flat_price:.2f} per kWh (current tariff, flat load)",
                                        f"emission factor {emission_factor} kg CO2e per kWh",
                                        f"repair cost {action_cost[0]:.0f} to {action_cost[1]:.0f} (assumed, not in data)"]})
    return sorted(actions, key=lambda a: a["payback_months"][1])


def what_if_tariff(daily: pd.DataFrame, period: str, compare_to: str, new_price_factor: float) -> dict:
    """Bill change if the current-quarter price rise had been scaled by new_price_factor (0 = no rise)."""
    ref = daily[quarter_mask(daily.index, compare_to)]
    cur = daily[quarter_mask(daily.index, period)]
    c0 = float(ref["cost"].sum())
    base = float(cur["cost"].sum())
    scenario = float(cur["cost_ref"].sum() + new_price_factor * (cur["cost"].sum() - cur["cost_ref"].sum()))
    return {"base_change_pct": round(100 * (base - c0) / c0, 2), "scenario_change_pct": round(100 * (scenario - c0) / c0, 2),
            "bill_difference": round(scenario - base, 2)}
