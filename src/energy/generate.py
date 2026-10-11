"""Synthetic data generator with a known true split of the bill change.

Energy is built from separate components (base load, production per product,
weather, leak, noise), so the cost of each component can be priced exactly and
the true contribution of every cause is known. Traps are then injected into the
files that are written to disk; truth.json always describes the clean data.

Usage: python -m energy.generate --out data --seeds 1 2 3 --control
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

START, MID, END = "2026-04-01", "2026-07-01", "2026-10-01"
PRODUCTS = {"A": 1.2, "B": 2.0, "C": 3.5}  # kWh per unit
SHARES = {"ref": [0.50, 0.30, 0.20], "cur": [0.46, 0.30, 0.24]}
UNITS_PER_DAY = {"ref": 3000.0, "cur": 3090.0}
BASE_KWH = {"work": 6000.0, "idle": 2500.0}
KWH_PER_CDD = 120.0
LEAK_KW, LEAK_FROM = 10.0, "2026-07-20"
HOLIDAYS = ["2026-05-01", "2026-06-17", "2026-08-15", "2026-09-04"]
HOLIDAY_LOGGED_AS_SHIFT = "2026-08-15"
TARIFF_CHANGE = "2026-08-15"
OLD_PRICE = {"peak": 8.0, "off_peak": 6.0}
NEW_PRICE = {"peak": 9.1, "off_peak": 6.8}
WRONG_NEW_OFF_PEAK = 7.4  # error in the first tariff file, fixed by the late drop
PEAK_HOURS = (6, 22)
MISCALIBRATED_METER, MISCALIBRATION = "line_2", 1.08
EMISSION_FACTOR_KG_PER_KWH = 0.4  # assumption, stated in the data card

SUB_SHARES = {
    "base": {"compressors": 0.22, "hvac": 0.15, "furnace": 0.18, "lighting": 0.14,
             "utilities": 0.14, "line_1": 0.04, "line_2": 0.04, "line_3": 0.04},
    "prod_A": {"line_1": 0.60, "furnace": 0.24, "compressors": 0.14},
    "prod_B": {"line_2": 0.60, "furnace": 0.24, "compressors": 0.14},
    "prod_C": {"line_3": 0.60, "furnace": 0.24, "compressors": 0.14},
    "weather": {"hvac": 1.0},
    "leak": {"compressors": 1.0},
}
SUB_METERS = ["compressors", "hvac", "furnace", "lighting", "utilities", "line_1", "line_2", "line_3"]


def _profile(lo_hour: int, hi_hour: int) -> np.ndarray:
    p = np.zeros(96)
    p[lo_hour * 4:hi_hour * 4] = 1.0
    return p / p.sum()


def _components(seed: int, leak: bool, nonlinear: bool = False):
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END, freq="D", inclusive="left")
    ts = pd.date_range(START, END, freq="15min", inclusive="left")
    n = len(days)
    cur = np.asarray(days >= pd.Timestamp(MID))
    holiday = np.asarray(days.isin(pd.to_datetime(HOLIDAYS)))
    work = (np.asarray(days.dayofweek) != 6) & ~holiday

    total_units = np.where(cur, UNITS_PER_DAY["cur"], UNITS_PER_DAY["ref"]) * rng.normal(1, 0.05, n)
    shares = np.vstack([rng.dirichlet(60 * np.array(SHARES["cur" if c else "ref"])) for c in cur])
    units = np.round(total_units[:, None] * shares * work[:, None]).astype(int)

    temp = np.where(cur, 27.5, 27.0) + rng.normal(0, 2.5, n)
    hot = np.asarray((days >= "2026-09-01") & (days < "2026-09-13"))
    temp = temp + 6.0 * hot
    cdd = np.maximum(temp - 22.0, 0.0)
    hdd = np.maximum(18.0 - temp, 0.0)

    flat, shift, afternoon = _profile(0, 24), _profile(*PEAK_HOURS), _profile(10, 20)
    base_work = 0.75 * flat + 0.25 * shift
    day_noise = rng.normal(1, 0.012, n)

    comp = {}
    comp["base"] = np.concatenate([
        (BASE_KWH["work"] * base_work if w else BASE_KWH["idle"] * flat) * f for w, f in zip(work, day_noise)])
    for j, (p, k) in enumerate(PRODUCTS.items()):
        comp[f"prod_{p}"] = np.concatenate([k * u * shift * f for u, f in zip(units[:, j], day_noise)])
    weather_kwh = KWH_PER_CDD * cdd + (6.0 * cdd ** 2 if nonlinear else 0.0)  # stress set: curved response
    comp["weather"] = np.concatenate([w * afternoon for w in weather_kwh])
    leak_kwh = LEAK_KW * 0.25 if leak else 0.0
    comp["leak"] = np.where(ts >= pd.Timestamp(LEAK_FROM), leak_kwh, 0.0)

    clean_total = sum(comp.values())
    main = clean_total * rng.normal(1, 0.03, len(ts))
    comp["noise"] = main - clean_total

    subs = {}
    for m in SUB_METERS:
        x = sum(comp[c] * s.get(m, 0.0) for c, s in SUB_SHARES.items())
        subs[m] = x * rng.normal(1, 0.01, len(ts))

    production = pd.DataFrame({
        "date": np.repeat(days.strftime("%Y-%m-%d"), len(PRODUCTS)),
        "product": np.tile(list(PRODUCTS), n),
        "units": units.ravel(),
        "shift_pattern": np.repeat(np.where(work, "2-shift", "off"), len(PRODUCTS)),
        "is_holiday": np.repeat(holiday, len(PRODUCTS)),
    })
    production.loc[production["date"] == HOLIDAY_LOGGED_AS_SHIFT, "shift_pattern"] = "2-shift"
    weather = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "avg_temp_c": temp.round(2),
                            "cooling_degree_days": cdd.round(2), "heating_degree_days": hdd.round(2)})
    return ts, days, comp, main, subs, production, weather, units, cur


def _tariff(new_off_peak: float) -> pd.DataFrame:
    rows = []
    for a, b, price in [(START, TARIFF_CHANGE, OLD_PRICE),
                        (TARIFF_CHANGE, END, {"peak": NEW_PRICE["peak"], "off_peak": new_off_peak})]:
        rows.append((a, b, "peak", price["peak"], PEAK_HOURS[0], PEAK_HOURS[1]))
        rows.append((a, b, "off_peak", price["off_peak"], PEAK_HOURS[1], PEAK_HOURS[0]))
    df = pd.DataFrame(rows, columns=["valid_from", "valid_to", "period", "price_per_kwh", "start_hour", "end_hour"])
    df["valid_from"] = df["valid_from"] + "T00:00:00Z"
    df["valid_to"] = df["valid_to"] + "T00:00:00Z"
    return df


def _truth(ts, comp, main, units, cur_day, new_off_peak: float, leak: bool) -> dict:
    is_peak = np.asarray((ts.hour >= PEAK_HOURS[0]) & (ts.hour < PEAK_HOURS[1]))
    p_old = np.where(is_peak, OLD_PRICE["peak"], OLD_PRICE["off_peak"])
    p_new = np.where(is_peak, NEW_PRICE["peak"], new_off_peak)
    p_act = np.where(np.asarray(ts >= pd.Timestamp(TARIFF_CHANGE)), p_new, p_old)
    cur = np.asarray(ts >= pd.Timestamp(MID))
    ref = ~cur

    c0 = float((p_act * main)[ref].sum())
    c1 = float((p_act * main)[cur].sum())
    pp = lambda x: 100.0 * x / c0
    tariff = float(((p_act - p_old) * main)[cur].sum())
    weather = float((p_old * comp["weather"])[cur].sum() - (p_old * comp["weather"])[ref].sum())
    waste = float((p_old * comp["leak"])[cur].sum())

    k = np.array(list(PRODUCTS.values()))
    u0, u1 = units[~cur_day].sum(axis=0), units[cur_day].sum(axis=0)
    big0, big1 = u0.sum(), u1.sum()
    k0, k1 = float(k @ u0 / big0), float(k @ u1 / big1)
    big0, big1 = float(big0), float(big1)
    mix = OLD_PRICE["peak"] * 0.5 * (big0 + big1) * (k1 - k0)  # production runs in peak hours only
    volume = (c1 - c0) - tariff - weather - waste - mix
    split = {"tariff": pp(tariff), "volume": float(pp(volume)), "product_mix": float(pp(mix)),
             "weather": pp(weather), "waste": pp(waste)}
    return {
        "period": "2026-Q3", "compare_to": "2026-Q2",
        "bill_reference": round(c0, 2), "bill_current": round(c1, 2),
        "total_change_pct": round(pp(c1 - c0), 4),
        "true_split_pp": {a: round(b, 4) for a, b in split.items()},
        "planted_waste": ([{"asset": "compressors", "pattern": "constant_leak", "from": LEAK_FROM,
                            "leak_kw": LEAK_KW, "saving_kwh_per_year": LEAK_KW * 8760}] if leak else []),
        "definitions": {
            "tariff": "(actual price - reference-quarter price) x current-quarter energy",
            "weather": "change in weather-driven energy at reference prices",
            "waste": "leak energy at reference prices",
            "product_mix": "Shapley share of the change in kWh per unit at reference prices",
            "volume": "remainder: output volume, calendar days and meter noise",
        },
    }


def _long(ts, main, subs) -> pd.DataFrame:
    frames = [pd.DataFrame({"meter_id": "main", "ts": ts, "kwh": main})]
    frames += [pd.DataFrame({"meter_id": m, "ts": ts, "kwh": v}) for m, v in subs.items()]
    df = pd.concat(frames, ignore_index=True)
    df["kwh"] = df["kwh"].round(4)
    return df


def _inject_traps(df: pd.DataFrame, rng) -> tuple[pd.DataFrame, dict]:
    ts = df["ts"]
    outage_all = (ts >= "2026-08-03 02:00") & (ts < "2026-08-03 08:00")
    outage_hvac = (df["meter_id"] == "hvac") & (ts >= "2026-05-20 13:00") & (ts < "2026-05-20 16:00")
    keep = ~(outage_all | outage_hvac)
    singles = rng.choice(df.index[keep], size=30, replace=False)
    keep.loc[singles] = False
    out = df[keep].copy()
    reset = out.index[(out["meter_id"] == "furnace") & (out["ts"] == pd.Timestamp("2026-06-10 03:15"))]
    out.loc[reset, "kwh"] = -round(float(out.loc[reset, "kwh"].iloc[0]) * 20, 4)
    dupes = out.loc[rng.choice(out.index, size=200, replace=False)]
    out = pd.concat([out, dupes]).sort_values(["meter_id", "ts"], kind="stable").reset_index(drop=True)
    traps = {"missing_rows": int((~keep).sum()), "duplicate_rows": 200, "meter_resets": 1,
             "outages": ["all meters 2026-08-03 02:00-08:00", "hvac 2026-05-20 13:00-16:00"]}
    return out, traps


def generate(seed: int, out_dir: Path, leak: bool = True, nonlinear: bool = False) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "late_drop").mkdir(parents=True, exist_ok=True)
    ts, days, comp, main, subs, production, weather, units, cur_day = _components(seed, leak, nonlinear)
    rng = np.random.default_rng(seed + 10_000)

    corrected = _long(ts, main, subs)
    raw = corrected.copy()
    mis = raw["meter_id"] == MISCALIBRATED_METER
    raw.loc[mis, "kwh"] = (raw.loc[mis, "kwh"] * MISCALIBRATION).round(4)
    raw, traps = _inject_traps(raw, rng)

    fmt = lambda d: d.assign(ts=d["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    fmt(raw).to_csv(out_dir / "meter_15min.csv", index=False)
    _tariff(WRONG_NEW_OFF_PEAK).to_csv(out_dir / "tariff.csv", index=False)
    production.to_csv(out_dir / "production.csv", index=False)
    weather.to_csv(out_dir / "weather.csv", index=False)

    fix = corrected[corrected["meter_id"] == MISCALIBRATED_METER]
    fmt(fix).to_csv(out_dir / "late_drop" / "meter_15min.csv", index=False)
    _tariff(NEW_PRICE["off_peak"]).to_csv(out_dir / "late_drop" / "tariff.csv", index=False)

    truth = {"seed": seed, "leak": leak,
             "as_loaded": _truth(ts, comp, main, units, cur_day, WRONG_NEW_OFF_PEAK, leak),
             "after_late_drop": _truth(ts, comp, main, units, cur_day, NEW_PRICE["off_peak"], leak),
             "emission_factor_kg_per_kwh": EMISSION_FACTOR_KG_PER_KWH}
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2))

    card = {
        "name": f"factory-energy-synthetic-seed-{seed}{'' if leak else '-control'}", "version": "1.0",
        "how_made": "python -m energy.generate (numpy generator, no LLM); seed fixed",
        "seed": seed,
        "contains": {"date_range": [START, END], "meters": ["main"] + SUB_METERS,
                     "row_counts": {"meter_15min": len(raw), "tariff": 4,
                                    "production": len(production), "weather": len(weather)}},
        "traps": {**traps, "sub_meters_below_main": "about 3% unmetered",
                  "tariff_change": TARIFF_CHANGE, "mix_shift": SHARES,
                  "hot_period": "2026-09-01 to 2026-09-12",
                  "holiday_logged_as_shift": HOLIDAY_LOGGED_AS_SHIFT,
                  "miscalibrated_meter": f"{MISCALIBRATED_METER} reads {MISCALIBRATION}x until the late drop",
                  "wrong_tariff": f"new off_peak {WRONG_NEW_OFF_PEAK} until the late drop",
                  "planted_leak": bool(leak)},
        "known_limits": ["intraday load shape is the same in both quarters, so no load shifting",
                         "curved weather response" if nonlinear else "linear weather response", "one factory, two quarters"],
        "privacy": "fully synthetic; no real company or personal data",
        "emission_factor_kg_per_kwh": EMISSION_FACTOR_KG_PER_KWH,
    }
    (out_dir / "data_card.json").write_text(json.dumps(card, indent=2))
    return truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--control", action="store_true", help="also write a dataset with no leak")
    ap.add_argument("--stress", action="store_true", help="also write a dataset the linear baseline cannot fit exactly")
    args = ap.parse_args()
    for s in args.seeds:
        t = generate(s, Path(args.out) / f"seed_{s}")
        print(f"seed_{s}: total {t['as_loaded']['total_change_pct']:.2f}%  {t['as_loaded']['true_split_pp']}")
    if args.control:
        t = generate(99, Path(args.out) / "control", leak=False)
        print(f"control: total {t['as_loaded']['total_change_pct']:.2f}%  {t['as_loaded']['true_split_pp']}")
    if args.stress:
        t = generate(7, Path(args.out) / "stress", nonlinear=True)
        print(f"stress: total {t['as_loaded']['total_change_pct']:.2f}%  {t['as_loaded']['true_split_pp']}")


if __name__ == "__main__":
    main()
