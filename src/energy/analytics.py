"""Deterministic analytics: pricing, baseline, bill bridge, asset use, waste, savings, what-if.

Every number an answer may quote is produced here. Nothing in this module calls an LLM.
"""
from __future__ import annotations

import os
from itertools import permutations

import numpy as np
import pandas as pd

from .pipeline import working_days

CAUSES = ["tariff", "volume", "product_mix", "weather", "waste"]
NIGHT_HOURS = (0, 5)  # used by the step-change detector: the quietest hours of the night
SHIFT_HOURS = tuple(int(h) for h in os.getenv("SHIFT_HOURS", "6,22").split(","))  # production shift
RECENT_DAYS = 28  # load profiles for charts show the last four weeks: what the factory does now

# Expected load outside the shift, as a share of the asset's load during the shift (low, high).
# Engineering rules of thumb, stated as assumptions; anything above the high end is avoidable.
OFF_HOURS_BENCHMARK = {
    "compressors": (0.05, 0.10),  # pressure kept up for a dryer or standby
    "lighting": (0.05, 0.15),     # security lighting
    "hvac": (0.05, 0.20),         # server room and frost protection
    "furnace": (0.20, 0.50),      # holding temperature overnight is normal
    "utilities": (0.60, 1.00),    # pumps and services run around the clock
    "line": (0.03, 0.15),         # controls on standby
}
DEFAULT_BENCHMARK = (0.10, 0.30)

# What to do about off-hours load on each kind of asset. Costs are assumed ranges in the data
# currency; the data pack has no cost table, so payback is only as good as these numbers.
ACTION_LIBRARY = {
    "compressors": {"action": "Find and fix compressed-air leaks (ultrasonic leak survey)",
                    "cost": (40_000, 80_000), "ease": "easy", "always_on": True},
    "lighting": {"action": "Switch hall lighting off after the last shift (timers or occupancy sensors)",
                 "cost": (120_000, 250_000), "ease": "easy"},
    "hvac": {"action": "Schedule HVAC off on non-working days (BMS time schedule)",
             "cost": (10_000, 25_000), "ease": "easy"},
    "line": {"action": "Power down {asset} standby loads between shifts",
             "cost": (20_000, 50_000), "ease": "easy"},
    "furnace": {"action": "Review furnace holding temperature and idle schedule",
                "cost": (50_000, 150_000), "ease": "medium"},
}
DEFAULT_ACTION = {"action": "Investigate off-hours load on {asset}", "cost": (25_000, 75_000), "ease": "medium"}
EASE_RANK = {"easy": 0, "medium": 1, "hard": 2}
DISPLAY_NAME = {"hvac": "HVAC", "compressors": "compressor"}


def _kind(asset: str) -> str:
    return "line" if asset.startswith("line") else asset


# ---- pricing and the daily table -------------------------------------------------

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
    daily["cdd"], daily["hdd"] = daily["cdd"] * daily["work"], daily["hdd"] * daily["work"]  # buildings are conditioned on working days
    daily.attrs["unit_cols"] = unit_cols
    return daily


# ---- baseline and the bill bridge ----------------------------------------------------

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
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss if ss > 0 else 0.0
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
        bridge.append({"cause": c, "pct_points": round(pts, 2), "range": [round(float(min(lo, pts)), 2), round(float(max(hi, pts)), 2)],
                       "amount": round(point["amounts"][c], 2)})
    total = 100 * (point["c1"] - c0) / c0
    # displayed values must add up too: put the rounding remainder on the largest cause
    gap = round(round(total, 2) - sum(b["pct_points"] for b in bridge), 2)
    if gap:
        big = max(bridge, key=lambda b: abs(b["pct_points"]))
        big["pct_points"] = round(big["pct_points"] + gap, 2)
        big["range"] = [min(big["range"][0], big["pct_points"]), max(big["range"][1], big["pct_points"])]
    fit = point["fit"]
    waste = next(b for b in bridge if b["cause"] == "waste")
    return {"status": "ok", "period": period, "compare_to": compare_to,
            "bill_reference": round(c0, 2), "bill_current": round(point["c1"], 2),
            "bill_change": round(point["c1"] - c0, 2),
            "total_change_pct": round(total, 2), "bridge": bridge,
            "uncontrollable_pct_points": round(sum(b["pct_points"] for b in bridge if b["cause"] != "waste"), 2),
            "waste_confirmed": waste["range"][0] > 0,
            "baseline": {"model": model, "r2": round(fit["r2"], 4), "coefficients": {k: round(v, 4) for k, v in fit["beta"].items()},
                         "target": "daily cost at reference-quarter prices", "trained_on": compare_to},
            "assumptions": ["reference-quarter tariff schedule is the 'old' price for every interval",
                            "calendar effect (number of working days) is counted in volume",
                            "ranges are the 5th to 95th percentile of 200 bootstrap refits of the baseline"]}


# ---- where the energy goes -------------------------------------------------------

def asset_breakdown(meters: pd.DataFrame, period: str, compare_to: str | None = None) -> list[dict]:
    q = meters[quarter_mask(meters.index, period)]
    main = float(q["main"].sum())
    subs = q.drop(columns="main").sum().sort_values(ascending=False)
    ref = meters[quarter_mask(meters.index, compare_to)] if compare_to else None
    rows = []
    for m, v in subs.items():
        row = {"asset": m, "kwh": round(float(v), 1), "share_of_main": round(float(v) / main, 4), "basis": "sub-meter"}
        if ref is not None and len(ref):
            r = float(ref[m].sum())
            row["change_pct"] = round(100 * (float(v) - r) / r, 1) if r else None
        rows.append(row)
    rows.append({"asset": "unmetered", "kwh": round(main - float(subs.sum()), 1),
                 "share_of_main": round(1 - float(subs.sum()) / main, 4), "basis": "estimated remainder"})
    return rows


def _masks(index: pd.DatetimeIndex, frames: dict) -> dict:
    work = working_days(frames["production"]).reindex(index.normalize()).fillna(False).to_numpy(bool)
    hour = np.asarray(index.hour)
    in_shift = (hour >= SHIFT_HOURS[0]) & (hour < SHIFT_HOURS[1])
    return {"work": work, "on": work & in_shift, "night": ~in_shift, "idle_day": ~work & in_shift}


def idle_load(meters: pd.DataFrame, frames: dict, period: str) -> list[dict]:
    """Energy used when nothing is produced: outside the shift on working days, and all day on
    non-working days. Reported by asset, with the hours it happens."""
    q = meters[quarter_mask(meters.index, period)]
    m = _masks(q.index, frames)
    idle = ~m["on"]
    days = q.index.normalize()
    n_idle_days = int(pd.Index(days[~m["work"]]).nunique())
    out = []
    for asset in q.columns.drop("main"):
        s = q[asset]
        nights, offdays = float(s[m["night"] & m["work"]].sum()), float(s[~m["work"]].sum())
        by_hour = (s[idle].groupby(s[idle].index.hour).mean() * 4).round(1)
        peak_hour = int(by_hour.idxmax()) if len(by_hour) else None
        out.append({"asset": asset, "idle_kwh": round(nights + offdays, 1),
                    "nights_on_working_days_kwh": round(nights, 1), "non_working_days_kwh": round(offdays, 1),
                    "idle_share_of_asset": round((nights + offdays) / float(s.sum()), 3) if s.sum() else 0.0,
                    "idle_avg_kw": round(float(s[idle].mean() * 4), 1), "busiest_idle_hour": peak_hour,
                    "non_working_days": n_idle_days})
    return sorted(out, key=lambda r: -r["idle_kwh"])


# ---- waste ---------------------------------------------------------------------------

def detect_leaks(meters: pd.DataFrame, frames: dict, period: str, compare_to: str,
                 min_rise: float = 0.10, z: float = 4.0, run: int = 5) -> list[dict]:
    """New waste: a sub-meter whose night-time load steps up and stays up for at least `run` days.
    Night load is compared with the reference quarter, within the same day type."""
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


def off_hours_waste(meters: pd.DataFrame, frames: dict, period: str, leaks: list[dict] | None = None) -> list[dict]:
    """Controllable waste: load outside the shift that is higher than the asset needs.

    For each sub-meter, compare its average load (a) at night and (b) during shift hours on
    non-working days with its load during the shift on working days. Load above the benchmark
    share is avoidable; the benchmark range gives the saving range. It is measured over the whole
    period, or from the day a step change started when one was found (`leaks`)."""
    whole = meters[quarter_mask(meters.index, period)]
    if whole.empty:
        return []
    all_days = whole.index.normalize()
    work_all = working_days(frames["production"]).reindex(pd.Index(all_days.unique())).fillna(False)
    idle_days_per_year = 365.0 * float((~work_all.astype(bool)).sum()) / max(len(work_all), 1)
    night_hours_per_day = 24 - (SHIFT_HOURS[1] - SHIFT_HOURS[0])
    since = {l["asset"]: l for l in (leaks or [])}
    findings = []
    for asset in whole.columns.drop("main"):
        q = whole[whole.index >= pd.Timestamp(since[asset]["from"])] if asset in since else whole
        m = _masks(q.index, frames)
        start, end = q.index.min().strftime("%Y-%m-%d"), q.index.max().strftime("%Y-%m-%d")
        kind = _kind(asset)
        lo_share, hi_share = OFF_HOURS_BENCHMARK.get(kind, DEFAULT_BENCHMARK)
        on_kw = float(q.loc[m["on"], asset].mean() * 4)
        if not on_kw or np.isnan(on_kw):
            continue
        always_on = ACTION_LIBRARY.get(kind, {}).get("always_on", False)
        windows = [("nights", m["night"], night_hours_per_day * 365.0,
                    f"{SHIFT_HOURS[1]:02d}:00-{SHIFT_HOURS[0]:02d}:00 every night")]
        if not always_on:
            windows.append(("non-working days", m["idle_day"], (SHIFT_HOURS[1] - SHIFT_HOURS[0]) * idle_days_per_year,
                            f"{SHIFT_HOURS[0]:02d}:00-{SHIFT_HOURS[1]:02d}:00 on Sundays and holidays"))
        for window, mask, hours_per_year, when in windows:
            if mask.sum() < 8:
                continue
            off_kw = float(q.loc[mask, asset].mean() * 4)
            excess = [off_kw - hi_share * on_kw, off_kw - lo_share * on_kw]
            if excess[0] < max(1.0, 0.03 * on_kw):  # not clearly above what the asset needs
                continue
            hours = 8760.0 if always_on else hours_per_year
            if asset in since:  # the size of the step change is a second, tighter measurement
                step = since[asset]["excess_kw_range"]
                excess = [max(excess[0], step[0]), max(excess[1], step[1])]
            ratio = (off_kw / on_kw) / hi_share
            findings.append({
                "asset": asset, "window": "day and night (always on)" if always_on else window, "when": "24/7" if always_on else when,
                "off_hours_kw": round(off_kw, 1), "on_shift_kw": round(on_kw, 1),
                "off_hours_share": round(off_kw / on_kw, 3), "benchmark_share": [lo_share, hi_share],
                "excess_kw": [round(excess[0], 1), round(excess[1], 1)], "hours_per_year": round(hours),
                "excess_kwh_per_year": [round(excess[0] * hours), round(excess[1] * hours)],
                "confidence": "high" if ratio >= 2 else "medium" if ratio >= 1.3 else "low",
                "measured_on": [start, end],
                "evidence_refs": [f"meter_15min:{asset}:{start}..{end}:{'00-24h' if always_on else window.replace(' ', '_')}"],
            })
    return sorted(findings, key=lambda f: -f["excess_kwh_per_year"][1])


def _window_price(frames: dict, latest: pd.Timestamp, window: str) -> float:
    idx = pd.date_range(latest.normalize(), periods=96, freq="15min")
    p = price_series(idx, frames["tariff"])
    hour = np.asarray(idx.hour)
    in_shift = (hour >= SHIFT_HOURS[0]) & (hour < SHIFT_HOURS[1])
    sel = {"nights": ~in_shift, "non-working days": in_shift}.get(window, np.ones(96, bool))
    return float(np.nanmean(p[sel]))


def estimate_savings(findings: list[dict], leaks: list[dict], meters: pd.DataFrame, frames: dict,
                     emission_factor: float) -> list[dict]:
    """Turn each waste finding into an action with ranges, ranked by payback then ease."""
    latest = meters.index.max()
    new_waste = {l["asset"]: l for l in leaks}
    actions = []
    for f in findings:
        lib = ACTION_LIBRARY.get(_kind(f["asset"]), DEFAULT_ACTION)
        price = _window_price(frames, latest, f["window"])
        kwh = f["excess_kwh_per_year"]
        money = [round(k * price) for k in kwh]
        cost = lib["cost"]
        payback = [round(12 * cost[0] / money[1], 1), round(12 * cost[1] / money[0], 1)] if money[0] > 0 else [None, None]
        name = DISPLAY_NAME.get(f["asset"], f["asset"].replace("_", " "))
        rationale = (f"{name[0].upper() + name[1:]} load is {f['off_hours_kw']} kW {('around the clock' if f['when'] == '24/7' else 'during ' + f['when'])}, "
                     f"{round(100 * f['off_hours_share'])}% of its load during the shift, where at most about "
                     f"{round(100 * f['benchmark_share'][1])}% would be expected.")
        if f["asset"] in new_waste:
            l = new_waste[f["asset"]]
            rationale += (f" Its night load rose from {l['reference_night_kw']} kW to {round(l['reference_night_kw'] + l['excess_kw'], 1)} kW "
                          f"on {l['from']} and has stayed there.")
        actions.append({
            "action": lib["action"].format(asset=f["asset"].replace("_", " ")), "asset": f["asset"], "window": f["window"],
            "saving_kwh_per_year": kwh, "saving_money_per_year": money,
            "saving_co2e_t_per_year": [round(k * emission_factor / 1000, 1) for k in kwh],
            "payback_months": payback, "implementation_cost": list(cost), "ease": lib["ease"],
            "rationale": rationale, "evidence_refs": f["evidence_refs"] + (new_waste[f["asset"]]["evidence_refs"] if f["asset"] in new_waste else []),
            "confidence": "high" if f["asset"] in new_waste else f["confidence"], "requires_approval": True,
            "is_new_waste": f["asset"] in new_waste,
            "assumptions": [f"price {price:.2f} per kWh in that window (current tariff)",
                            f"emission factor {emission_factor} kg CO2e per kWh",
                            f"implementation cost {cost[0]:,} to {cost[1]:,} (assumed, not in data)",
                            f"expected off-hours load {round(100 * f['benchmark_share'][0])}-{round(100 * f['benchmark_share'][1])}% of shift load (rule of thumb)"]})
    actions.sort(key=lambda a: (a["payback_months"][1] if a["payback_months"][1] is not None else 1e9, EASE_RANK.get(a["ease"], 9)))
    for i, a in enumerate(actions, 1):
        a["rank"] = i
    return actions


def savings_summary(actions: list[dict]) -> dict:
    if not actions:
        return {"actions": 0, "saving_money_per_year": [0, 0], "saving_kwh_per_year": [0, 0], "saving_co2e_t_per_year": [0, 0],
                "implementation_cost": [0, 0]}
    tot = lambda k: [round(sum(a[k][0] for a in actions), 1), round(sum(a[k][1] for a in actions), 1)]
    return {"actions": len(actions), "saving_money_per_year": tot("saving_money_per_year"),
            "saving_kwh_per_year": tot("saving_kwh_per_year"), "saving_co2e_t_per_year": tot("saving_co2e_t_per_year"),
            "implementation_cost": tot("implementation_cost"),
            "fastest_payback_months": actions[0]["payback_months"]}


def asset_check(meters: pd.DataFrame, frames: dict, asset: str, period: str, compare_to: str,
                findings: list[dict], leaks: list[dict]) -> dict:
    """Evidence for or against blaming one asset: its share, its change, and any waste found on it."""
    cur, ref = meters[quarter_mask(meters.index, period)], meters[quarter_mask(meters.index, compare_to)]
    k1, k0 = float(cur[asset].sum()), float(ref[asset].sum())
    m1, m0 = float(cur["main"].sum()), float(ref["main"].sum())
    return {"asset": asset, "kwh": round(k1, 1), "share_of_main": round(k1 / m1, 4),
            "change_pct": round(100 * (k1 - k0) / k0, 1) if k0 else None,
            "factory_change_pct": round(100 * (m1 - m0) / m0, 1),
            "off_hours_waste": [f for f in findings if f["asset"] == asset],
            "new_waste": [l for l in leaks if l["asset"] == asset]}


# ---- profiles for charts ------------------------------------------------------------

def load_profile(meters: pd.DataFrame, frames: dict, period: str, asset: str, recent_days: int = RECENT_DAYS) -> dict:
    """Average kW by hour of day (working vs non-working days), a weekday x hour heatmap, and
    the night load per day across all the data (to see when a step change started)."""
    q = meters[quarter_mask(meters.index, period)]
    q = q[q.index >= q.index.max().normalize() - pd.Timedelta(days=recent_days - 1)]
    m = _masks(q.index, frames)
    s = q[asset] * 4
    hourly = lambda mask: [round(float(v), 1) for v in s[mask].groupby(s[mask].index.hour).mean().reindex(range(24)).fillna(0)]
    heat = s.groupby([s.index.dayofweek, s.index.hour]).mean().unstack().reindex(index=range(7), columns=range(24))
    whole = meters[asset] * 4
    night = whole[(whole.index.hour >= NIGHT_HOURS[0]) & (whole.index.hour < NIGHT_HOURS[1])]
    nightly = night.groupby(night.index.normalize()).mean()
    lo, hi = OFF_HOURS_BENCHMARK.get(_kind(asset), DEFAULT_BENCHMARK)
    on_kw = float(s[m["on"]].mean())
    return {"asset": asset, "hours": list(range(24)), "working_day_kw": hourly(m["work"]), "non_working_day_kw": hourly(~m["work"]),
            "expected_off_hours_kw": [round(lo * on_kw, 1), round(hi * on_kw, 1)], "shift_hours": list(SHIFT_HOURS),
            "heatmap_kw": [[round(float(v), 1) if pd.notna(v) else None for v in row] for row in heat.to_numpy()],
            "nightly_kw": {"dates": [d.strftime("%Y-%m-%d") for d in nightly.index], "kw": [round(float(v), 1) for v in nightly]},
            "measured_on": [q.index.min().strftime("%Y-%m-%d"), q.index.max().strftime("%Y-%m-%d")]}


def daily_series(daily: pd.DataFrame) -> dict:
    return {"dates": [d.strftime("%Y-%m-%d") for d in daily.index], "kwh": [round(float(v)) for v in daily["kwh"]],
            "cost": [round(float(v)) for v in daily["cost"]], "cdd": [round(float(v), 1) for v in daily["cdd"]],
            "working": [bool(v) for v in daily["work"]]}


# ---- what-if --------------------------------------------------------------------------

def what_if_tariff(daily: pd.DataFrame, period: str, compare_to: str, rise_factor: float | None = None,
                   price_factor: float | None = None) -> dict:
    """Bill change under a different current-quarter tariff: either the price rise scaled by
    rise_factor (0 = no rise), or every current price scaled by price_factor (0.9 = 10% lower)."""
    ref = daily[quarter_mask(daily.index, compare_to)]
    cur = daily[quarter_mask(daily.index, period)]
    c0, base = float(ref["cost"].sum()), float(cur["cost"].sum())
    if price_factor is not None:
        scenario = base * price_factor
    else:
        scenario = float(cur["cost_ref"].sum() + (rise_factor or 0.0) * (base - cur["cost_ref"].sum()))
    return {"base_change_pct": round(100 * (base - c0) / c0, 2), "scenario_change_pct": round(100 * (scenario - c0) / c0, 2),
            "bill_current": round(base), "bill_scenario": round(scenario), "bill_difference": round(scenario - base)}


def what_if_actions(daily: pd.DataFrame, period: str, compare_to: str, actions: list[dict]) -> dict:
    """Bill change if the recommended actions had been in place for the whole current quarter."""
    ref = daily[quarter_mask(daily.index, compare_to)]
    cur = daily[quarter_mask(daily.index, period)]
    c0, base = float(ref["cost"].sum()), float(cur["cost"].sum())
    share = len(cur) / 365.0
    saved = [sum(a["saving_money_per_year"][i] for a in actions) * share for i in (0, 1)]
    return {"modelled_actions": [a["action"] for a in actions], "base_change_pct": round(100 * (base - c0) / c0, 2),
            "scenario_change_pct": [round(100 * (base - saved[1] - c0) / c0, 2), round(100 * (base - saved[0] - c0) / c0, 2)],
            "quarter_saving": [round(saved[0]), round(saved[1])]}
