import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy import analytics as an  # noqa: E402
from energy.generate import generate  # noqa: E402
from energy.pipeline import apply_ingest, clean, load_dir, parse_table, validate  # noqa: E402

PERIOD, REF = "2026-Q3", "2026-Q2"


@pytest.fixture(scope="module")
def waste_set(tmp_path_factory):
    d = tmp_path_factory.mktemp("waste")
    return d, generate(11, d)


@pytest.fixture(scope="module")
def control_set(tmp_path_factory):
    d = tmp_path_factory.mktemp("control")
    return d, generate(12, d, waste=False)


def _run(d, late=False):
    frames = load_dir(d, late_drop=late)
    meters, quality = clean(frames)
    daily = an.daily_table(meters, frames, REF)
    return frames, meters, quality, daily


def test_files_follow_the_annex_b_schema(waste_set):
    d, _ = waste_set
    cols = {name: list(pd.read_csv(d / f"{name}.csv", nrows=1).columns) for name in ("meter_15min", "production", "weather")}
    assert cols["meter_15min"] == ["meter_id", "ts", "kwh"]
    assert cols["production"] == ["date", "product", "units", "shift_pattern", "is_holiday"]
    assert cols["weather"] == ["date", "avg_temp_c", "cooling_degree_days", "heating_degree_days"]
    assert {"valid_from", "valid_to", "period", "price_per_kwh"} <= set(pd.read_csv(d / "tariff.csv").columns)


def test_cleaning_leaves_a_complete_non_negative_grid(waste_set):
    _, meters, quality, _ = _run(waste_set[0])
    assert not meters.isna().any().any()
    assert (meters >= 0).all().all()
    assert len(meters) == 183 * 96
    assert quality["duplicates_removed"] == 200 and quality["negative_readings"] == 1


def test_validation_flags_the_holiday_logged_as_a_shift(waste_set):
    checks = {c["id"]: c for c in validate(load_dir(waste_set[0]))}
    assert not checks[5]["passed"] and "2026-08-15" in checks[5]["detail"]
    assert all(checks[i]["passed"] for i in (1, 2, 3, 4, 6))


def test_bad_rows_are_rejected_not_fatal():
    df = pd.DataFrame({"meter_id": ["main", "main"], "ts": ["2026-04-01T00:00:00Z", "not a time"], "kwh": [1.0, 2.0]})
    kept, rejected = parse_table("meter_15min", df)
    assert len(kept) == 1 and rejected == 1


@pytest.mark.parametrize("late", [False, True])
def test_bridge_adds_up_and_is_close_to_truth(waste_set, late):
    d, truth = waste_set
    b = an.decompose(_run(d, late)[3], PERIOD, REF, n_boot=50)
    parts = {x["cause"]: x["pct_points"] for x in b["bridge"]}
    assert abs(sum(parts.values()) - b["total_change_pct"]) < 0.005
    for cause, true in truth["after_late_drop" if late else "as_loaded"]["true_split_pp"].items():
        assert abs(parts[cause] - true) < 0.35, cause
    assert all(x["range"][0] <= x["pct_points"] <= x["range"][1] for x in b["bridge"])


def test_new_waste_is_the_compressor_step_change(waste_set):
    frames, meters, *_ = _run(waste_set[0])
    leaks = an.detect_leaks(meters, frames, PERIOD, REF)
    assert [l["asset"] for l in leaks] == ["compressors"]
    assert abs(pd.Timestamp(leaks[0]["from"]) - pd.Timestamp("2026-07-20")) <= pd.Timedelta(days=2)


def test_every_planted_waste_is_found_and_its_saving_is_inside_the_range(waste_set):
    d, truth = waste_set
    frames, meters, *_ = _run(d, late=True)
    leaks = an.detect_leaks(meters, frames, PERIOD, REF)
    actions = an.estimate_savings(an.off_hours_waste(meters, frames, PERIOD, leaks), leaks, meters, frames, 0.71)
    found = {a["asset"]: a for a in actions}
    for w in truth["after_late_drop"]["planted_waste"]:
        lo, hi = found[w["asset"]]["saving_kwh_per_year"]
        assert lo <= w["saving_kwh_per_year"] <= hi, w["asset"]
    assert set(found) == {w["asset"] for w in truth["after_late_drop"]["planted_waste"]}


def test_actions_are_ranked_by_payback_and_need_approval(waste_set):
    frames, meters, *_ = _run(waste_set[0])
    leaks = an.detect_leaks(meters, frames, PERIOD, REF)
    actions = an.estimate_savings(an.off_hours_waste(meters, frames, PERIOD, leaks), leaks, meters, frames, 0.71)
    paybacks = [a["payback_months"][1] for a in actions]
    assert paybacks == sorted(paybacks) and [a["rank"] for a in actions] == list(range(1, len(actions) + 1))
    for a in actions:
        assert a["requires_approval"] is True and a["evidence_refs"]
        for k in ("saving_kwh_per_year", "saving_money_per_year", "saving_co2e_t_per_year", "payback_months"):
            assert a[k][0] <= a[k][1]


def test_no_false_waste_claims_on_the_control_set(control_set):
    frames, meters, _, daily = _run(control_set[0])
    assert an.detect_leaks(meters, frames, PERIOD, REF) == []
    assert an.off_hours_waste(meters, frames, PERIOD) == []
    b = an.decompose(daily, PERIOD, REF, n_boot=50)
    assert abs(next(x for x in b["bridge"] if x["cause"] == "waste")["pct_points"]) < 0.3


def test_numeric_answer_matches_independent_calculation(waste_set):
    _, meters, *_ = _run(waste_set[0])
    raw = pd.read_csv(waste_set[0] / "meter_15min.csv")
    raw = raw[(raw.meter_id == "lighting") & raw.ts.between("2026-07-01", "2026-10-01")].drop_duplicates(["meter_id", "ts"])
    ours = next(a for a in an.asset_breakdown(meters, PERIOD) if a["asset"] == "lighting")["kwh"]
    assert ours == pytest.approx(raw.kwh.sum(), rel=0.005)  # differs only by the gap-filled intervals


def test_mid_quarter_tariff_is_priced_by_interval(waste_set):
    frames = load_dir(waste_set[0])
    idx = pd.DatetimeIndex(["2026-08-14 12:00", "2026-08-15 12:00", "2026-08-15 23:00"])
    assert an.price_series(idx, frames["tariff"]).tolist() == [8.0, 9.1, 7.4]
    assert an.price_series(idx, frames["tariff"], as_of=pd.Timestamp("2026-04-01")).tolist() == [8.0, 8.0, 6.0]


def test_late_drop_changes_the_tariff_effect_only(waste_set):
    d, truth = waste_set
    before = an.decompose(_run(d)[3], PERIOD, REF, n_boot=20)
    after = an.decompose(_run(d, late=True)[3], PERIOD, REF, n_boot=20)
    get = lambda b, c: next(x["pct_points"] for x in b["bridge"] if x["cause"] == c)
    assert get(after, "tariff") < get(before, "tariff")
    assert abs(get(after, "weather") - get(before, "weather")) < 0.02
    assert abs(after["total_change_pct"] - truth["after_late_drop"]["total_change_pct"]) < 0.1


def test_ingest_merges_safely(waste_set):
    frames = load_dir(waste_set[0])
    old = frames["meter_15min"]
    fix, _ = parse_table("meter_15min", pd.read_csv(waste_set[0] / "late_drop" / "meter_15min.csv"))
    once, how = apply_ingest("meter_15min", old, fix)
    twice, _ = apply_ingest("meter_15min", once, fix)
    assert "line_2" in how and len(once) == len(twice)  # a full series for one meter replaces that meter only
    one_day = old[(old.meter_id == "main") & (old.ts.dt.strftime("%Y-%m-%d") == "2026-09-30")]
    merged, how = apply_ingest("meter_15min", old, one_day)
    assert "updated" in how and (merged.meter_id == "main").sum() == (old.drop_duplicates(["meter_id", "ts"]).meter_id == "main").sum()


def test_insufficient_data_is_reported(waste_set):
    assert an.decompose(_run(waste_set[0])[3].iloc[:40], PERIOD, REF)["status"] == "insufficient_data"
