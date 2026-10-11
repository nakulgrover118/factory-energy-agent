import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy import analytics as an  # noqa: E402
from energy.generate import generate  # noqa: E402
from energy.pipeline import apply_ingest, clean, load_dir, parse_table, validate  # noqa: E402

PERIOD, REF = "2026-Q3", "2026-Q2"


@pytest.fixture(scope="module")
def leak_set(tmp_path_factory):
    d = tmp_path_factory.mktemp("leak")
    return d, generate(11, d)


@pytest.fixture(scope="module")
def control_set(tmp_path_factory):
    d = tmp_path_factory.mktemp("control")
    return d, generate(12, d, leak=False)


def _run(d, late=False):
    frames = load_dir(d, late_drop=late)
    meters, quality = clean(frames)
    daily = an.daily_table(meters, frames, REF)
    return frames, meters, quality, daily


def test_cleaning_leaves_a_complete_non_negative_grid(leak_set):
    frames, meters, quality, _ = _run(leak_set[0])
    assert not meters.isna().any().any()
    assert (meters >= 0).all().all()
    assert len(meters) == 183 * 96
    assert quality["duplicates_removed"] == 200 and quality["negative_readings"] == 1


def test_validation_flags_the_holiday_logged_as_a_shift(leak_set):
    checks = {c["id"]: c for c in validate(load_dir(leak_set[0]))}
    assert not checks[5]["passed"] and "2026-08-15" in checks[5]["detail"]
    assert all(checks[i]["passed"] for i in (1, 2, 3, 4, 6))


def test_bad_rows_are_rejected_not_fatal():
    df = pd.DataFrame({"meter_id": ["main", "main"], "ts": ["2026-04-01T00:00:00Z", "not a time"], "kwh": [1.0, 2.0]})
    kept, rejected = parse_table("meter_15min", df)
    assert len(kept) == 1 and rejected == 1


def test_bridge_adds_up_and_is_close_to_truth(leak_set):
    d, truth = leak_set
    *_, daily = _run(d)
    b = an.decompose(daily, PERIOD, REF, n_boot=50)
    parts = {x["cause"]: x["pct_points"] for x in b["bridge"]}
    assert abs(sum(parts.values()) - b["total_change_pct"]) < 0.005
    for cause, true in truth["as_loaded"]["true_split_pp"].items():
        assert abs(parts[cause] - true) < 0.5, cause
    assert all(x["range"][0] <= x["pct_points"] <= x["range"][1] for x in b["bridge"])


def test_leak_found_on_the_right_asset_and_saving_range_covers_truth(leak_set):
    d, truth = leak_set
    frames, meters, *_ = _run(d)
    leaks = an.detect_leaks(meters, frames, PERIOD, REF)
    assert [l["asset"] for l in leaks] == ["compressors"]
    action = an.estimate_savings(leaks, meters, frames, 0.4)[0]
    lo, hi = action["saving_kwh_per_year"]
    assert lo <= truth["as_loaded"]["planted_waste"][0]["saving_kwh_per_year"] <= hi
    assert action["requires_approval"] is True


def test_no_false_waste_claim_on_control(control_set):
    frames, meters, _, daily = _run(control_set[0])
    assert an.detect_leaks(meters, frames, PERIOD, REF) == []
    waste = next(x for x in an.decompose(daily, PERIOD, REF, n_boot=50)["bridge"] if x["cause"] == "waste")
    assert abs(waste["pct_points"]) < 0.4


def test_numeric_answer_matches_independent_calculation(leak_set):
    frames, meters, *_ = _run(leak_set[0])
    raw = pd.read_csv(leak_set[0] / "meter_15min.csv")
    raw = raw[(raw.meter_id == "lighting") & raw.ts.between("2026-07-01", "2026-10-01")].drop_duplicates(["meter_id", "ts"])
    ours = next(a for a in an.asset_breakdown(meters, PERIOD) if a["asset"] == "lighting")["kwh"]
    assert ours == pytest.approx(raw.kwh.sum(), rel=0.005)  # differs only by the gap-filled intervals


def test_mid_quarter_tariff_is_priced_by_interval(leak_set):
    frames = load_dir(leak_set[0])
    idx = pd.DatetimeIndex(["2026-08-14 12:00", "2026-08-15 12:00", "2026-08-15 23:00"])
    assert an.price_series(idx, frames["tariff"]).tolist() == [8.0, 9.1, 7.4]
    assert an.price_series(idx, frames["tariff"], as_of=pd.Timestamp("2026-04-01")).tolist() == [8.0, 8.0, 6.0]


def test_late_drop_changes_tariff_effect_only(leak_set):
    d, truth = leak_set
    before = an.decompose(_run(d)[3], PERIOD, REF, n_boot=20)
    after = an.decompose(_run(d, late=True)[3], PERIOD, REF, n_boot=20)
    get = lambda b, c: next(x["pct_points"] for x in b["bridge"] if x["cause"] == c)
    assert get(after, "tariff") < get(before, "tariff")
    assert abs(get(after, "weather") - get(before, "weather")) < 0.02
    assert abs(after["total_change_pct"] - truth["after_late_drop"]["total_change_pct"]) < 0.1


def test_ingest_is_safe_to_repeat(leak_set):
    frames = load_dir(leak_set[0])
    new, _ = parse_table("meter_15min", pd.read_csv(leak_set[0] / "late_drop" / "meter_15min.csv"))
    once = apply_ingest("meter_15min", frames["meter_15min"], new)
    twice = apply_ingest("meter_15min", once, new)
    assert len(once) == len(twice)


def test_insufficient_data_is_reported(leak_set):
    daily = _run(leak_set[0])[3]
    assert an.decompose(daily.iloc[:40], PERIOD, REF)["status"] == "insufficient_data"
