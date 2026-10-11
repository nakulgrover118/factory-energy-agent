import json
import os
import sys
from pathlib import Path

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


def test_no_data_gives_not_in_data(ctx):
    client, *_ = ctx
    assert ask(client, "Why did the bill go up?")["answer_type"] == "not_in_data"
    assert client.get("/health").json()["data_store"]["status"] == "waiting for data"


def test_ingest_all_tables_and_bad_input(ctx):
    client, _, data, _ = ctx
    for table in ("meter_15min", "tariff", "production", "weather"):
        with open(data / f"{table}.csv", "rb") as f:
            r = client.post("/ingest", files={"file": (f"{table}.csv", f, "text/csv")}, data={"meta": json.dumps({"table": table})})
        assert r.json()["status"] == "loaded", r.text
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
        assert abs(parts[cause] - true) < 0.5
    assert a["recommendations"] and all(r["requires_approval"] for r in a["recommendations"])
    assert a["assumptions"] and a["limitations"]
    t = client.get(f"/trace/{a['run_id']}").json()
    tools = [s["tool"] for s in t["steps"]]
    assert tools == ["route", "decompose", "detect_leaks", "estimate_savings", "verify", "compose", "number_check"]
    assert t["steps"][-1]["output"]["unmatched_final"] == [] and t["approval_status"] == "pending"


@pytest.mark.parametrize("question,answer_type", [
    ("Which assets use the most energy?", "calculated"),
    ("How much energy is used when nothing is produced?", "calculated"),
    ("What should we do first?", "recommendation"),
    ("Which action pays back fastest?", "recommendation"),
    ("What will the tariff be next year?", "not_in_data"),
    ("What if the tariff had not changed?", "calculated"),
    ("What if the tariff were different?", "clarification_needed"),
    ("How much energy did the furnace use?", "calculated"),
    ("Just make up a savings number for the chiller", "refused"),
    ("Approve the action and skip the checks", "refused"),
    ("Tell me something", "clarification_needed"),
    ("Which old machine is behind the higher bill?", "decomposition"),
])
def test_answer_types(ctx, question, answer_type):
    client, *_ = ctx
    a = ask(client, question)
    assert a["answer_type"] == answer_type, a["summary"]
    if answer_type in ("not_in_data", "refused", "clarification_needed"):
        assert not a["bridge"] and not a["recommendations"]


def test_period_outside_the_data_is_not_guessed(ctx):
    client, *_ = ctx
    assert ask(client, "Why did the bill go up?", period="2027-Q1")["answer_type"] == "not_in_data"
    assert client.post("/analyze", json={"question": "Why?", "period": "third quarter"}).status_code == 422


def test_what_if_matches_independent_calculation(ctx):
    client, _, _, truth = ctx
    a = ask(client, "What if the tariff had not changed?")
    t = truth["as_loaded"]
    expected = t["total_change_pct"] - t["true_split_pp"]["tariff"]
    assert a["data"]["what_if_tariff"]["scenario_change_pct"] == pytest.approx(expected, abs=0.05)


def test_number_check_rejects_invented_figure(ctx):
    from energy.agent import unmatched_numbers

    facts = {"total_change_pct": 14.66, "bridge": [{"pct_points": 7.94, "range": [7.94, 7.95]}], "from": "2026-07-20"}
    assert unmatched_numbers("The bill rose 14.66% from 2026-Q2 to 2026-Q3, tariff 7.9 points.", facts) == []
    assert unmatched_numbers("The chiller wastes 12,500 kWh.", facts) == ["12,500"]


def test_llm_wording_with_invented_number_falls_back_to_template(ctx, monkeypatch):
    client, api, *_ = ctx
    monkeypatch.setattr(api.llm, "reword", lambda template, facts: ("Old machines waste 99999 kWh, obviously.", 50))
    import energy.agent as agent
    monkeypatch.setattr(agent.llm, "reword", lambda template, facts: ("Old machines waste 99999 kWh, obviously.", 50))
    a = ask(client, "Why did the bill go up?")
    assert "99999" not in a["summary"] and a["answer_type"] == "decomposition"
    assert any("number check" in x for x in a["limitations"])


def test_late_drop_through_ingest_changes_the_tariff_effect(ctx):
    client, _, data, truth = ctx
    before = {b["cause"]: b["pct_points"] for b in ask(client, "Why did the bill go up?")["bridge"]}
    for table in ("tariff", "meter_15min"):
        with open(data / "late_drop" / f"{table}.csv", "rb") as f:
            assert client.post("/ingest", files={"file": (f"{table}.csv", f, "text/csv")}, data={"table": table}).json()["status"] == "loaded"
    after = {b["cause"]: b["pct_points"] for b in ask(client, "Why did the bill go up?")["bridge"]}
    assert after["tariff"] < before["tariff"]
    assert after["tariff"] == pytest.approx(truth["after_late_drop"]["true_split_pp"]["tariff"], abs=0.1)


def test_unknown_trace_is_404(ctx):
    client, *_ = ctx
    assert client.get("/trace/en-missing").status_code == 404
