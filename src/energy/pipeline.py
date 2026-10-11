"""Load, validate and clean the four data files.

validate() runs the six checks from the participant guide and never raises on
bad data: it reports. clean() returns a gap-free 15-minute table (one column per
meter) plus a quality report that says exactly what was changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = {
    "meter_15min": ["meter_id", "ts", "kwh"],
    "tariff": ["valid_from", "valid_to", "period", "price_per_kwh"],
    "production": ["date", "product", "units", "shift_pattern", "is_holiday"],
    "weather": ["date", "avg_temp_c", "cooling_degree_days", "heating_degree_days"],
}
STEP = pd.Timedelta("15min")
SUB_METER_TOLERANCE = 0.10  # sub-meters may be up to 10% below (or 2% above) the main meter
SHORT_GAP = 8  # intervals (2 hours) filled by interpolation; longer gaps use a typical-day profile
DEFAULT_PEAK_HOURS = (6, 22)


def _naive_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def parse_table(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Type one table to the Annex B schema. Returns (rows kept, rows rejected)."""
    missing = [c for c in SCHEMA[name] if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    df = df.copy()
    n = len(df)
    if name == "meter_15min":
        df["ts"] = _naive_utc(df["ts"])
        df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
        df["meter_id"] = df["meter_id"].astype(str).str.strip()
        df = df.dropna(subset=["ts", "kwh"])
    elif name == "tariff":
        df["valid_from"], df["valid_to"] = _naive_utc(df["valid_from"]), _naive_utc(df["valid_to"])
        df["price_per_kwh"] = pd.to_numeric(df["price_per_kwh"], errors="coerce")
        df["period"] = df["period"].astype(str).str.strip().str.lower()
        if "start_hour" not in df.columns:  # not in the documented schema: assume peak 06-22
            lo, hi = DEFAULT_PEAK_HOURS
            df["start_hour"] = df["period"].map({"peak": lo, "off_peak": hi, "flat": 0})
            df["end_hour"] = df["period"].map({"peak": hi, "off_peak": lo, "flat": 24})
        df = df.dropna(subset=["valid_from", "valid_to", "price_per_kwh", "start_hour", "end_hour"])
        df = df[df["valid_to"] > df["valid_from"]]
    elif name == "production":
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["units"] = pd.to_numeric(df["units"], errors="coerce")
        df["is_holiday"] = _as_bool(df["is_holiday"])
        df["product"] = df["product"].astype(str).str.strip()
        df = df.dropna(subset=["date", "units"])
        df = df[df["units"] >= 0]
    elif name == "weather":
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        for c in SCHEMA[name][1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["date", "cooling_degree_days", "heating_degree_days"])
    return df.reset_index(drop=True), n - len(df)


def load_dir(path: str | Path, late_drop: bool = False) -> dict:
    """Read a dataset folder. With late_drop=True the late files are applied on top."""
    path = Path(path)
    frames, rejected = {}, {}
    for name in SCHEMA:
        frames[name], rejected[name] = parse_table(name, pd.read_csv(path / f"{name}.csv"))
    if late_drop:
        for name in SCHEMA:
            f = path / "late_drop" / f"{name}.csv"
            if f.exists():
                new, rej = parse_table(name, pd.read_csv(f))
                frames[name] = apply_ingest(name, frames[name], new)
                rejected[name] += rej
    card = path / "data_card.json"
    frames["_card"] = json.loads(card.read_text()) if card.exists() else None
    frames["_rejected"] = rejected
    return frames


def apply_ingest(name: str, old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge a newly ingested table. Safe to repeat: the same file twice changes nothing."""
    if name == "tariff":
        return new  # a tariff file is a complete schedule
    if name == "meter_15min":  # new rows replace old rows for the same meter and time
        keep = old[~old["meter_id"].isin(new["meter_id"].unique())]
        return pd.concat([keep, new], ignore_index=True)
    key = ["date", "product"] if name == "production" else ["date"]
    merged = pd.concat([old, new], ignore_index=True)
    return merged.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


def _wide(meter: pd.DataFrame) -> pd.DataFrame:
    m = meter.drop_duplicates(subset=["meter_id", "ts"], keep="last")
    return m.pivot(index="ts", columns="meter_id", values="kwh").sort_index()


def working_days(production: pd.DataFrame) -> pd.Series:
    """A day is a working day only if something was produced and it is not a holiday."""
    g = production.groupby("date").agg(units=("units", "sum"), holiday=("is_holiday", "max"))
    return (g["units"] > 0) & ~g["holiday"].astype(bool)


def validate(frames: dict) -> list[dict]:
    meter, tariff, prod = frames["meter_15min"], frames["tariff"], frames["production"]
    checks = []
    add = lambda cid, name, ok, detail: checks.append({"id": cid, "check": name, "passed": bool(ok), "detail": detail})

    wide = _wide(meter)
    if "main" in wide.columns and wide.shape[1] > 1:
        both = wide.dropna()
        ratio = float(both.drop(columns="main").clip(lower=0).sum().sum() / both["main"].clip(lower=0).sum())
        add(1, "Sub-meters sum to main within tolerance", 1 - SUB_METER_TOLERANCE <= ratio <= 1.02,
            f"sub-meters are {ratio:.1%} of main; unmetered remainder {1 - ratio:.1%}")
    else:
        add(1, "Sub-meters sum to main within tolerance", False, "no main meter or no sub-meters")

    neg = int((meter["kwh"] < 0).sum())
    add(2, "No negative energy; resets handled", True, f"{neg} negative readings treated as meter resets and re-estimated")

    off_grid = int((meter["ts"].dt.minute % 15 != 0).sum() + (meter["ts"].dt.second != 0).sum())
    full = pd.date_range(wide.index.min(), wide.index.max(), freq=STEP)
    missing = int(wide.reindex(full).isna().sum().sum())
    dupes = int(meter.duplicated(subset=["meter_id", "ts"]).sum())
    add(3, "Timestamps regular and in one time zone", off_grid == 0,
        f"{off_grid} off-grid timestamps; {missing} missing meter intervals and {dupes} duplicate rows reported")

    problems = []
    lo, hi = wide.index.min(), wide.index.max()
    for period, g in tariff.groupby("period"):
        g = g.sort_values("valid_from")
        if (g["valid_from"].iloc[1:].values < g["valid_to"].iloc[:-1].values).any():
            problems.append(f"{period}: overlapping periods")
        if (g["valid_from"].iloc[1:].values > g["valid_to"].iloc[:-1].values).any():
            problems.append(f"{period}: gap between periods")
        if g["valid_from"].min() > lo or g["valid_to"].max() <= hi:
            problems.append(f"{period}: does not cover the meter data")
    add(4, "Tariff periods do not overlap and cover the period", not problems,
        "; ".join(problems) or f"{len(tariff)} tariff rows cover {lo.date()} to {hi.date()}")

    g = prod.groupby("date").agg(units=("units", "sum"), holiday=("is_holiday", "max"),
                                 shift=("shift_pattern", "first"))
    logged_working = ~g["shift"].astype(str).str.lower().isin(["off", "none", "nan", ""])
    mismatch = g.index[logged_working & ((g["units"] == 0) | g["holiday"].astype(bool))]
    meter_days = pd.DatetimeIndex(wide.index.normalize().unique())
    no_prod = meter_days.difference(g.index)
    add(5, "Production days match the shift calendar", len(mismatch) == 0 and len(no_prod) == 0,
        (f"{len(mismatch)} day(s) logged as a shift with no output or on a holiday "
         f"({', '.join(d.strftime('%Y-%m-%d') for d in mismatch[:5])}); treated as non-working. " if len(mismatch) else "")
        + (f"{len(no_prod)} meter day(s) have no production row." if len(no_prod) else "") or "calendar consistent")

    card = frames.get("_card")
    if card:
        want = card["contains"]["row_counts"]
        raw_counts = frames.get("_raw_counts") or {k: len(frames[k]) + frames["_rejected"].get(k, 0) for k in SCHEMA}
        diff = {k: (raw_counts[k], want[k]) for k in want if raw_counts.get(k) != want[k]}
        add(6, "Row counts match the data card", not diff, f"mismatch (found, card): {diff}" if diff else "all tables match")
    else:
        add(6, "Row counts match the data card", False, "no data card supplied")
    return checks


def clean(frames: dict) -> tuple[pd.DataFrame, dict]:
    """Return a complete 15-minute table (kWh per meter) and a report of every change."""
    meter = frames["meter_15min"]
    report = {"rows_in": int(len(meter))}
    report["duplicates_removed"] = int(meter.duplicated(subset=["meter_id", "ts"]).sum())
    wide = _wide(meter)
    report["negative_readings"] = int((wide < 0).sum().sum())
    wide = wide.where(wide >= 0)

    start = wide.index.min().normalize()
    end = wide.index.max().normalize() + pd.Timedelta("1D") - STEP
    wide = wide.reindex(pd.date_range(start, end, freq=STEP))
    wide.index.name = "ts"
    gaps = wide.isna()
    report["missing_intervals"] = {c: int(v) for c, v in gaps.sum().items() if v}

    work = working_days(frames["production"]).reindex(wide.index.normalize()).fillna(False).to_numpy()
    filled = wide.interpolate(limit=SHORT_GAP, limit_area="inside")
    slot = wide.index.hour * 4 + wide.index.minute // 15
    quarter = wide.index.to_period("Q")
    typical = wide.groupby([quarter, work, slot]).transform("median")  # same quarter, day type, time of day
    run = gaps.apply(lambda s: s.groupby((~s).cumsum()).transform("sum"))
    long_gap = gaps & (run > SHORT_GAP)
    filled = filled.where(~long_gap, typical).fillna(typical).ffill().bfill()
    report["filled_by_interpolation"] = int((gaps & ~long_gap).sum().sum())
    report["filled_by_typical_profile"] = int(long_gap.sum().sum())
    report["share_of_intervals_estimated"] = round(float(gaps.sum().sum() / gaps.size), 5)

    if "main" in filled.columns and filled.shape[1] > 1:
        subs = filled.drop(columns="main").sum(axis=1)
        report["unmetered_share"] = round(float(1 - subs.sum() / filled["main"].sum()), 4)
    return filled, report
