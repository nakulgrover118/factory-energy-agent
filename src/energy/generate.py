"""Synthetic data generator with a known true split of the bill change.

Writes the four CSV files in the Annex B schema (meter_15min, tariff, production,
weather), a late-drop folder (corrected tariff and one recalibrated sub-meter), a
private truth.json and a data card (JSON and Markdown).

Energy is built per asset from separate components, so the cost of every cause and
the size of every waste pattern is known exactly:
  base load      each asset's legitimate load during the shift and outside it
  production     kWh per unit of each product, on its line, the furnace and the compressors
  weather        cooling energy in the afternoon, proportional to cooling degree-days
  waste          three controllable patterns (see WASTE)
  noise          3% on the main meter, 1% on each sub-meter

Usage: python -m energy.generate --out data --seeds 1 2 3 --control --stress
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

START, MID, END = "2026-04-01", "2026-07-01", "2026-10-01"
SHIFT = (6, 22)  # two shifts, 06:00-22:00; also the peak tariff window
PRODUCTS = {"A": 1.2, "B": 2.0, "C": 3.5}  # kWh per unit
PRODUCT_LINE = {"A": "line_1", "B": "line_2", "C": "line_3"}
PRODUCTION_SPLIT = {"line": 0.60, "furnace": 0.24, "compressors": 0.14}  # the other 2% is unmetered
SHARES = {"ref": [0.50, 0.30, 0.20], "cur": [0.46, 0.30, 0.24]}  # product mix shifts towards C
UNITS_PER_DAY = {"ref": 3000.0, "cur": 3090.0}
BASE_KW = {  # legitimate load: (during the shift on a working day, at any other time)
    "compressors": (40.0, 6.0), "hvac": (60.0, 10.0), "furnace": (30.0, 30.0), "lighting": (40.0, 4.0),
    "utilities": (15.0, 15.0), "line_1": (5.0, 2.0), "line_2": (5.0, 2.0), "line_3": (5.0, 2.0),
}
UNMETERED_KW = 12.0  # offices and canteen: on the main meter only
KWH_PER_CDD = 120.0
WASTE = {
    "compressor_leak": {"asset": "compressors", "kw": 10.0, "from": "2026-07-20",
                        "what": "compressed-air leak, running day and night"},
    "lights_on_at_night": {"asset": "lighting", "kw": 30.0,
                           "what": "hall lighting left on after the last shift (22:00-06:00)"},
    "hvac_on_non_working_days": {"asset": "hvac", "kw": 50.0,
                                 "what": "HVAC running and cooling 06:00-22:00 on Sundays and holidays"},
}
HOLIDAYS = ["2026-05-01", "2026-06-17", "2026-08-15", "2026-09-04"]
HOLIDAY_LOGGED_AS_SHIFT = "2026-08-15"
TARIFF_CHANGE = "2026-08-15"
OLD_PRICE = {"peak": 8.0, "off_peak": 6.0}  # INR per kWh
NEW_PRICE = {"peak": 9.1, "off_peak": 6.8}
WRONG_NEW_OFF_PEAK = 7.4  # error in the first tariff file, fixed by the late drop
MISCALIBRATED_METER, MISCALIBRATION = "line_2", 1.08
EMISSION_FACTOR_KG_PER_KWH = 0.71  # assumed grid average, stated in the data card
CURRENCY = "INR"
SUB_METERS = list(BASE_KW)


def _components(seed: int, waste: bool, nonlinear: bool):
    """Per-interval kWh for every (component, asset) pair. asset None = unmetered."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END, freq="D", inclusive="left")
    ts = pd.date_range(START, END, freq="15min", inclusive="left")
    n = len(days)
    day = np.repeat(np.arange(n), 96)
    hour = np.asarray(ts.hour)
    cur = np.asarray(days >= pd.Timestamp(MID))
    holiday = np.asarray(days.isin(pd.to_datetime(HOLIDAYS)))
    work = (np.asarray(days.dayofweek) != 6) & ~holiday
    in_shift = (hour >= SHIFT[0]) & (hour < SHIFT[1])
    on = work[day] & in_shift

    total_units = np.where(cur, UNITS_PER_DAY["cur"], UNITS_PER_DAY["ref"]) * rng.normal(1, 0.05, n)
    shares = np.vstack([rng.dirichlet(60 * np.array(SHARES["cur" if c else "ref"])) for c in cur])
    units = np.round(total_units[:, None] * shares * work[:, None]).astype(int)

    temp = np.where(cur, 27.5, 27.0) + rng.normal(0, 2.5, n)
    hot = np.asarray((days >= "2026-09-01") & (days < "2026-09-13"))
    temp = temp + 6.0 * hot
    cdd = np.maximum(temp - 22.0, 0.0)
    hdd = np.maximum(18.0 - temp, 0.0)

    comp: dict[tuple[str, str | None], np.ndarray] = {}
    day_noise = rng.normal(1, 0.012, n)[day]
    for asset, (kw_on, kw_off) in BASE_KW.items():
        comp[("base", asset)] = np.where(on, kw_on, kw_off) * 0.25 * day_noise
    comp[("base", None)] = np.full(len(ts), UNMETERED_KW * 0.25)

    shift_slots = int(in_shift[:96].sum())
    for j, (p, k) in enumerate(PRODUCTS.items()):
        e = np.where(in_shift, (k * units[:, j])[day] / shift_slots, 0.0)
        comp[(f"prod_{p}", PRODUCT_LINE[p])] = e * PRODUCTION_SPLIT["line"]
        comp[(f"prod_{p}", "furnace")] = e * PRODUCTION_SPLIT["furnace"]
        comp[(f"prod_{p}", "compressors")] = e * PRODUCTION_SPLIT["compressors"]
        comp[(f"prod_{p}", None)] = e * (1 - sum(PRODUCTION_SPLIT.values()))

    afternoon = (hour >= 10) & (hour < 20)
    cooling = KWH_PER_CDD * cdd + (6.0 * cdd ** 2 if nonlinear else 0.0)  # stress set: curved response
    cooling_kwh = np.where(afternoon, cooling[day] / int(afternoon[:96].sum()), 0.0)
    comp[("weather", "hvac")] = np.where(work[day], cooling_kwh, 0.0)  # the building is cooled on working days

    zero = np.zeros(len(ts))
    w = WASTE
    comp[("compressor_leak", "compressors")] = (
        np.where(np.asarray(ts >= pd.Timestamp(w["compressor_leak"]["from"])), w["compressor_leak"]["kw"] * 0.25, 0.0)
        if waste else zero)
    comp[("lights_on_at_night", "lighting")] = (
        np.where(~in_shift, w["lights_on_at_night"]["kw"] * 0.25, 0.0) if waste else zero)
    comp[("hvac_on_non_working_days", "hvac")] = (  # left on: it runs and cools an empty building
        np.where(~work[day] & in_shift, w["hvac_on_non_working_days"]["kw"] * 0.25 + cooling_kwh, 0.0) if waste else zero)

    clean_total = sum(comp.values())
    main = clean_total * rng.normal(1, 0.03, len(ts))
    subs = {m: sum(v for (c, a), v in comp.items() if a == m) * rng.normal(1, 0.01, len(ts)) for m in SUB_METERS}

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
    ctx = {"ts": ts, "days": days, "day": day, "work": work, "in_shift": in_shift, "cur_day": cur}
    return ctx, comp, main, subs, production, weather, units


def _tariff(new_off_peak: float) -> pd.DataFrame:
    rows = []
    for a, b, price in [(START, TARIFF_CHANGE, OLD_PRICE),
                        (TARIFF_CHANGE, END, {"peak": NEW_PRICE["peak"], "off_peak": new_off_peak})]:
        rows.append((a, b, "peak", price["peak"], SHIFT[0], SHIFT[1]))
        rows.append((a, b, "off_peak", price["off_peak"], SHIFT[1], SHIFT[0]))
    df = pd.DataFrame(rows, columns=["valid_from", "valid_to", "period", "price_per_kwh", "start_hour", "end_hour"])
    df["valid_from"] = df["valid_from"] + "T00:00:00Z"
    df["valid_to"] = df["valid_to"] + "T00:00:00Z"
    return df


def _truth(ctx, comp, main, units, new_off_peak: float, waste: bool) -> dict:
    ts = ctx["ts"]
    is_peak = np.asarray((ts.hour >= SHIFT[0]) & (ts.hour < SHIFT[1]))
    p_old = np.where(is_peak, OLD_PRICE["peak"], OLD_PRICE["off_peak"])
    p_new = np.where(is_peak, NEW_PRICE["peak"], new_off_peak)
    p_act = np.where(np.asarray(ts >= pd.Timestamp(TARIFF_CHANGE)), p_new, p_old)
    cur = np.asarray(ts >= pd.Timestamp(MID))
    ref = ~cur

    c0, c1 = float((p_act * main)[ref].sum()), float((p_act * main)[cur].sum())
    pp = lambda x: 100.0 * x / c0
    tariff = float(((p_act - p_old) * main)[cur].sum())
    weather = float((p_old * comp[("weather", "hvac")])[cur].sum() - (p_old * comp[("weather", "hvac")])[ref].sum())
    # waste in the bridge is the change in waste energy between the quarters (mostly the new leak);
    # patterns present in both quarters also show up as savings opportunities
    waste_kwh = sum(comp[(w, WASTE[w]["asset"])] for w in WASTE)
    new_waste = float((p_old * waste_kwh)[cur].sum() - (p_old * waste_kwh)[ref].sum())

    k = np.array(list(PRODUCTS.values()))
    cur_day = ctx["cur_day"]
    u0, u1 = units[~cur_day].sum(axis=0), units[cur_day].sum(axis=0)
    big0, big1 = float(u0.sum()), float(u1.sum())
    k0, k1 = float(k @ u0 / big0), float(k @ u1 / big1)
    mix = OLD_PRICE["peak"] * 0.5 * (big0 + big1) * (k1 - k0)  # production runs in peak hours only
    volume = (c1 - c0) - tariff - weather - new_waste - mix
    split = {"tariff": pp(tariff), "volume": pp(volume), "product_mix": pp(mix), "weather": pp(weather), "waste": pp(new_waste)}

    n_days = len(ctx["days"])
    idle_interval = ~ctx["work"][ctx["day"]] & ctx["in_shift"]
    hvac_idle_kwh = float(comp[("hvac_on_non_working_days", "hvac")][idle_interval].sum())
    planted = []
    if waste:
        w = WASTE
        planted = [
            {"asset": "compressors", "pattern": "compressor_leak", "from": w["compressor_leak"]["from"],
             "kw": w["compressor_leak"]["kw"], "saving_kwh_per_year": w["compressor_leak"]["kw"] * 8760},
            {"asset": "lighting", "pattern": "lights_on_at_night", "kw": w["lights_on_at_night"]["kw"],
             "saving_kwh_per_year": w["lights_on_at_night"]["kw"] * (24 - (SHIFT[1] - SHIFT[0])) * 365},
            {"asset": "hvac", "pattern": "hvac_on_non_working_days", "kw": w["hvac_on_non_working_days"]["kw"],
             "saving_kwh_per_year": round(hvac_idle_kwh * 365 / n_days)},  # includes cooling on those days
        ]
    return {
        "period": "2026-Q3", "compare_to": "2026-Q2",
        "bill_reference": round(c0, 2), "bill_current": round(c1, 2),
        "total_change_pct": round(pp(c1 - c0), 4),
        "true_split_pp": {a: round(b, 4) for a, b in split.items()},
        "planted_waste": planted,
        "definitions": {
            "tariff": "(actual price - reference-quarter price) x current-quarter energy",
            "weather": "change in cooling energy at reference prices",
            "waste": "change in energy used by the planted waste patterns, at reference prices",
            "product_mix": "Shapley share of the change in kWh per unit at reference prices",
            "volume": "remainder: output volume, calendar days and meter noise",
            "saving_kwh_per_year": "energy the planted pattern wastes in a year; the true saving if it is fixed",
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


def _card_markdown(card: dict, truth: dict) -> str:
    rc = card["contains"]["row_counts"]
    waste = "\n".join(f"- {WASTE[w['pattern']]['what']} on `{w['asset']}`" for w in truth["after_late_drop"]["planted_waste"]) or "- none (control set)"
    return f"""# Data card: {card['name']}

| Field | Value |
| --- | --- |
| Name and version | {card['name']}, version {card['version']} |
| How it was made | `python -m energy.generate`, numpy random generator, seed {card['seed']}. No LLM and no real data. |
| Date range | {START} to {END} (exclusive), two quarters, UTC, 15-minute steps |
| Tables and rows | meter_15min {rc['meter_15min']:,}; tariff {rc['tariff']}; production {rc['production']:,}; weather {rc['weather']} |
| Meters | main plus {len(SUB_METERS)} sub-meters: {', '.join(SUB_METERS)} |
| Currency | {CURRENCY} per kWh |
| Emission factor | {EMISSION_FACTOR_KG_PER_KWH} kg CO2e per kWh (assumed grid average) |
| Privacy | Fully synthetic. No real company or personal data. |

## How the energy is built

Each asset has a legitimate load during the shift (06:00-22:00 on working days) and outside it, plus
production energy (kWh per unit by product: {', '.join(f'{p} {k}' for p, k in PRODUCTS.items())}), cooling energy
proportional to cooling degree-days ({KWH_PER_CDD:g} kWh per degree-day) and the waste patterns below.
The main meter carries everything plus {UNMETERED_KW:g} kW of unmetered load and 3% noise; sub-meters carry 1% noise.

## Controllable waste planted on purpose

{waste}

## Traps and edge cases

- Tariff rises on {TARIFF_CHANGE}, in the middle of Q3 (peak {OLD_PRICE['peak']} to {NEW_PRICE['peak']}, off-peak {OLD_PRICE['off_peak']} to {NEW_PRICE['off_peak']}).
- The first tariff file has a wrong new off-peak price ({WRONG_NEW_OFF_PEAK}); the late drop corrects it.
- `{MISCALIBRATED_METER}` reads {MISCALIBRATION}x too high until the late drop recalibrates it.
- Product mix shifts towards product C (more kWh per unit) with no waste involved.
- Hot period 2026-09-01 to 2026-09-12 raises cooling.
- Public holiday {HOLIDAY_LOGGED_AS_SHIFT} is logged with a normal shift pattern.
- Sub-meters do not add up to the main meter (unmetered offices and canteen).
- {card['traps']['missing_rows']} missing rows (two outages and random gaps), {card['traps']['duplicate_rows']} duplicate rows, one meter reset (a negative reading on the furnace).

## Known limits

- One factory, two quarters; the factory clock is UTC.
- Intraday load shape does not change between quarters, so load shifting is not represented.
- {"Curved" if "curved" in " ".join(card['known_limits']) else "Linear"} weather response.
"""


def generate(seed: int, out_dir: Path, waste: bool = True, nonlinear: bool = False) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "late_drop").mkdir(parents=True, exist_ok=True)
    ctx, comp, main, subs, production, weather, units = _components(seed, waste, nonlinear)
    rng = np.random.default_rng(seed + 10_000)

    corrected = _long(ctx["ts"], main, subs)
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

    truth = {"seed": seed, "waste": waste,
             "as_loaded": _truth(ctx, comp, main, units, WRONG_NEW_OFF_PEAK, waste),
             "after_late_drop": _truth(ctx, comp, main, units, NEW_PRICE["off_peak"], waste),
             "emission_factor_kg_per_kwh": EMISSION_FACTOR_KG_PER_KWH}
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2))

    card = {
        "name": f"factory-energy-synthetic-seed-{seed}{'' if waste else '-control'}", "version": "2.0",
        "how_made": "python -m energy.generate (numpy generator, no LLM); seed fixed",
        "seed": seed,
        "contains": {"date_range": [START, END], "meters": ["main"] + SUB_METERS,
                     "row_counts": {"meter_15min": len(raw), "tariff": 4,
                                    "production": len(production), "weather": len(weather)}},
        "traps": {**traps, "sub_meters_below_main": "unmetered offices and canteen",
                  "tariff_change": TARIFF_CHANGE, "mix_shift": SHARES,
                  "hot_period": "2026-09-01 to 2026-09-12",
                  "holiday_logged_as_shift": HOLIDAY_LOGGED_AS_SHIFT,
                  "miscalibrated_meter": f"{MISCALIBRATED_METER} reads {MISCALIBRATION}x until the late drop",
                  "wrong_tariff": f"new off_peak {WRONG_NEW_OFF_PEAK} until the late drop",
                  "planted_waste": [w["pattern"] for w in truth["as_loaded"]["planted_waste"]]},
        "known_limits": ["intraday load shape is the same in both quarters, so no load shifting",
                         "curved weather response" if nonlinear else "linear weather response", "one factory, two quarters"],
        "privacy": "fully synthetic; no real company or personal data",
        "currency": CURRENCY,
        "emission_factor_kg_per_kwh": EMISSION_FACTOR_KG_PER_KWH,
    }
    (out_dir / "data_card.json").write_text(json.dumps(card, indent=2))
    (out_dir / "DATA_CARD.md").write_text(_card_markdown(card, truth), encoding="utf-8")
    return truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--control", action="store_true", help="also write a dataset with no waste")
    ap.add_argument("--stress", action="store_true", help="also write a dataset the linear baseline cannot fit exactly")
    args = ap.parse_args()
    for s in args.seeds:
        t = generate(s, Path(args.out) / f"seed_{s}")
        print(f"seed_{s}: total {t['as_loaded']['total_change_pct']:.2f}%  {t['as_loaded']['true_split_pp']}")
    if args.control:
        t = generate(99, Path(args.out) / "control", waste=False)
        print(f"control: total {t['as_loaded']['total_change_pct']:.2f}%  {t['as_loaded']['true_split_pp']}")
    if args.stress:
        t = generate(7, Path(args.out) / "stress", nonlinear=True)
        print(f"stress: total {t['as_loaded']['total_change_pct']:.2f}%  {t['as_loaded']['true_split_pp']}")


if __name__ == "__main__":
    main()
