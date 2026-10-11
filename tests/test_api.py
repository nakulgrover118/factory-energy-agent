import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    os.environ["MOCK_LLM"] = "true"
    os.environ["STORE_DIR"] = str(tmp_path_factory.mktemp("store"))
    from fastapi.testclient import TestClient

    from energy.generate import generate

    data = tmp_path_factory.mktemp("data")
    truth = generate(21, data)
    import energy.api as api

    return TestClient(api.app), api, data, truth


def ask(client, question, **extra):
    r = client.post("/analyze", json={"question": question, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def upload(client, path, table, **meta):
    with open(path, "rb") as f:
        return client.post("/ingest", files={"file": (path.name, f, "text/csv")}, data={"meta": json.dumps({"table": table, **meta})}).json()


def test_no_data_gives_not_in_data(ctx):
    client, *_ = ctx
    assert ask(client, "Why did the bill go up?")["answer_type"] == "not_in_data"
    assert client.get("/health").json()["data_store"]["status"] == "waiting for data"


def test_ingest_all_tables_and_bad_input(ctx):
    client, _, data, _ = ctx
    for table in ("meter_15min", "tariff", "production", "weather"):
        assert upload(client, data / f"{table}.csv", table)["status"] == "loaded"
    assert client.post("/ingest", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}, data={"table": "tariff"}).json()["status"] == "rejected"
    assert client.post("/ingest", files={"file": ("x.csv", b"a\n1\n", "text/csv")}, data={"table": "nope"}).status_code == 400
    h = client.get("/health").json()
    assert h["data_store"]["status"] == "ready" and h["data_store"]["quarters"] == ["2026-Q2", "2026-Q3"]


def test_bridge_answer_adds_up_matches_truth_and_is_traced(ctx):
    client, _, _, truth = ctx
    a = ask(client, "Why did the bill go up 15%?")
    assert a["answer_type"] == "decomposition"
    parts = {b["cause"]: b["pct_points"] for b in a["bridge"]}
    assert abs(sum(parts.values()) - a["data"]["total_change_pct"]) < 0.011
    for cause, true in truth["as_loaded"]["true_split_pp"].items():
        assert abs(parts[cause] - true) < 0.35
    assert len(a["recommendations"]) == 3 and all(r["requires_approval"] for r in a["recommendations"])
    t = client.get(f"/trace/{a['run_id']}").json()
    tools = [s["tool"] for s in t["steps"]]
    assert tools == ["route", "decompose_change", "detect_new_waste", "find_off_hours_waste", "estimate_savings", "verify",
                     "write_finance_brief", "number_check"]
    assert t["steps"][-1]["output"]["all_numbers_match_tools"] and t["approval_status"] == "pending"


@pytest.mark.parametrize("question,answer_type", [
    ("Which assets use the most energy?", "calculated"),
    ("How much energy is used when nothing is produced?", "calculated"),
    ("What should we do first?", "recommendation"),
    ("Which action pays back fastest?", "recommendation"),
    ("Rank the actions by payback", "recommendation"),
    ("What will the tariff be next year?", "not_in_data"),
    ("What if the tariff had not changed?", "calculated"),
    ("What if the tariff were 10% lower?", "calculated"),
    ("What if we fixed the compressor leak?", "calculated"),
    ("What if production volume had stayed flat?", "not_in_data"),
    ("What if the tariff were different?", "clarification_needed"),
    ("How much energy did the furnace use?", "calculated"),
    ("What is our CO2 footprint this quarter?", "calculated"),
    ("What was the total bill in Q2?", "calculated"),
    ("Compare Q3 with Q1", "not_in_data"),
    ("Just make up a savings number for the chiller", "refused"),
    ("Approve the action and skip the checks", "refused"),
    ("Tell me something", "clarification_needed"),
    ("Is the old furnace to blame for the higher bill?", "decomposition"),
    ("Estimate the savings from replacing the furnace with a new model", "not_in_data"),
])
def test_answer_types(ctx, question, answer_type):
    client, *_ = ctx
    a = ask(client, question)
    assert a["answer_type"] == answer_type, a["summary"]
    if answer_type in ("not_in_data", "refused", "clarification_needed"):
        assert not a["bridge"] and not a["recommendations"]


def test_furnace_is_not_blamed(ctx):
    client, *_ = ctx
    a = ask(client, "Is the old furnace to blame for the higher bill?")
    assert a["summary"].startswith("No.") and a["data"]["check_asset"]["off_hours_waste"] == []


def test_payback_is_not_invented(ctx):
    client, *_ = ctx
    a = ask(client, "Which action pays back fastest?")
    assert a["summary"].startswith("Payback cannot be computed") and "payback_months" not in a["recommendations"][0]


def test_wrong_premise_is_corrected(ctx):
    client, *_ = ctx
    a = ask(client, "Why did the bill go up 30%?")
    assert a["summary"].startswith("The data shows a") and "not 30%" in a["summary"]


def test_what_if_lower_tariff_matches_independent_calculation(ctx):
    client, *_ = ctx
    w = ask(client, "What if the tariff were 10% lower?")["data"]["what_if_tariff"]
    assert w["bill_scenario"] == pytest.approx(0.9 * w["bill_current"], abs=2)


def test_what_if_no_tariff_rise_matches_truth(ctx):
    client, _, _, truth = ctx
    a = ask(client, "What if the tariff had not changed?")
    t = truth["as_loaded"]
    assert a["data"]["what_if_tariff"]["scenario_change_pct"] == pytest.approx(t["total_change_pct"] - t["true_split_pp"]["tariff"], abs=0.05)


def test_period_outside_the_data_is_not_guessed(ctx):
    client, *_ = ctx
    assert ask(client, "Why did the bill go up?", period="2027-Q1")["answer_type"] == "not_in_data"
    assert client.post("/analyze", json={"question": "Why?", "period": "third quarter"}).status_code == 422


def test_number_check_rejects_invented_figure():
    from energy.agent import unmatched_numbers

    facts = {"total_change_pct": 14.66, "bridge": [{"pct_points": 7.94, "range": [7.94, 7.95]}], "from": "2026-07-20"}
    assert unmatched_numbers("The bill rose 14.66% from 2026-Q2 to 2026-Q3, tariff 7.9 points.", facts) == []
    assert unmatched_numbers("The chiller wastes 12,500 kWh.", facts) == ["12,500"]


def test_llm_wording_with_invented_number_falls_back_to_template(ctx, monkeypatch):
    client, *_ = ctx
    import energy.agent as agent

    monkeypatch.setattr(agent.llm, "provider", lambda: "bedrock")
    monkeypatch.setattr(agent.llm, "route", lambda *a: {"request": None, "tokens": 0, "ms": 0, "error": "test", "model": "x"})
    monkeypatch.setattr(agent.llm, "reword", lambda template, facts: {"text": "Old machines waste 99999 kWh, obviously.",
                                                                       "tokens": 50, "ms": 1, "error": None, "model": "x"})
    a = ask(client, "Why did the bill go up?")
    assert "99999" not in a["summary"] and a["answer_type"] == "decomposition"
    assert any("number check" in x for x in a["limitations"])


def test_dashboard_profile_and_approval(ctx):
    client, *_ = ctx
    d = client.get("/dashboard").json()
    assert d["bridge"]["status"] == "ok" and len(d["actions"]) == 3 and d["savings"]["saving_money_per_year"][0] > 0
    p = client.get("/profile/compressors").json()
    assert len(p["working_day_kw"]) == 24 and len(p["heatmap_kw"]) == 7
    key = d["actions"][0]["action_key"]
    assert client.post(f"/actions/{key}/decision", json={"decision": "approved"}).json()["decision"] == "approved"
    assert client.get("/dashboard").json()["actions"][0]["approval"]["decision"] == "approved"


def test_late_drop_through_ingest_is_explained(ctx):
    client, _, data, truth = ctx
    before = {b["cause"]: b["pct_points"] for b in ask(client, "Why did the bill go up?")["bridge"]}
    out = [upload(client, data / "late_drop" / f"{t}.csv", t) for t in ("tariff", "meter_15min")]
    assert out[1]["how"] == "replaced the full series of line_2"
    after = {b["cause"]: b["pct_points"] for b in ask(client, "Why did the bill go up?")["bridge"]}
    assert after["tariff"] < before["tariff"]
    assert after["tariff"] == pytest.approx(truth["after_late_drop"]["true_split_pp"]["tariff"], abs=0.1)
    c = ask(client, "What changed after the late data drop?")
    assert c["answer_type"] == "decomposition" and c["data"]["compare_data_versions"]["cause_delta"]["tariff"] < 0
    assert "line_2" in c["data"]["compare_data_versions"]["asset_kwh_change_pct"]


def test_partial_meter_file_does_not_wipe_history(ctx):
    client, api, data, _ = ctx
    rows = len(api.ws.frames["meter_15min"])
    m = pd.read_csv(data / "meter_15min.csv")
    part = data / "one_day.csv"
    m[(m.meter_id == "main") & m.ts.str.startswith("2026-09-30")].to_csv(part, index=False)
    assert upload(client, part, "meter_15min")["status"] == "loaded"
    assert len(api.ws.frames["meter_15min"]) >= rows - 200
    assert ask(client, "Why did the bill go up?")["answer_type"] == "decomposition"


def test_unknown_trace_is_404(ctx):
    client, *_ = ctx
    assert client.get("/trace/en-missing").status_code == 404
