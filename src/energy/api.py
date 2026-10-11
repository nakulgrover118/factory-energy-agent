"""FastAPI contract: POST /analyze, POST /ingest, GET /health, GET /trace/{run_id}, plus read-only
dashboard endpoints, manager approvals and an admin loader.

Run: uvicorn energy.api:app --app-dir src --port 8000
"""
from __future__ import annotations

import io
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .settings import load_env

load_env()

from . import analytics as an  # noqa: E402
from . import llm  # noqa: E402
from .agent import build_graph, new_run_id  # noqa: E402
from .pipeline import SCHEMA  # noqa: E402
from .store import Workspace  # noqa: E402

app = FastAPI(title="Factory Energy Cost Decomposition and Savings Agent", version="1.0.0")
ws = Workspace()
graph = build_graph(ws)


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    period: str | None = Field(default=None, pattern=r"^\d{4}-Q[1-4]$", examples=["2026-Q3"])
    compare_to: str | None = Field(default=None, pattern=r"^\d{4}-Q[1-4]$", examples=["2026-Q2"])


class BridgeItem(BaseModel):
    cause: str
    pct_points: float
    range: list[float]


class Recommendation(BaseModel):
    rank: int
    action: str
    asset: str
    rationale: str
    evidence_refs: list[str]
    confidence: str
    estimated_impact: str
    requires_approval: bool
    saving_kwh_per_year: list[float]
    saving_money_per_year: list[float]
    saving_co2e_t_per_year: list[float]
    ease: str
    action_key: str
    approval: dict[str, Any] = {}


class AnalyzeResponse(BaseModel):
    run_id: str
    answer_type: str
    summary: str
    bridge: list[BridgeItem] = []
    recommendations: list[Recommendation] = []
    data: dict[str, Any] = {}
    assumptions: list[str] = []
    limitations: list[str] = []


class Decision(BaseModel):
    decision: str = Field(pattern=r"^(approved|rejected|pending)$")
    note: str = Field(default="", max_length=500)


def _recommendation(a: dict) -> dict:
    keys = ("rank", "action", "asset", "rationale", "evidence_refs", "confidence", "requires_approval", "ease", "saving_kwh_per_year",
            "saving_money_per_year", "saving_co2e_t_per_year", "action_key", "approval")
    return {**{k: a[k] for k in keys},
            "estimated_impact": (f"Saves {a['saving_kwh_per_year'][0]:,} to {a['saving_kwh_per_year'][1]:,} kWh and "
                                 f"{a['saving_money_per_year'][0]:,} to {a['saving_money_per_year'][1]:,} {ws.currency()} a year")}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> dict:
    run_id, t0 = new_run_id(), time.perf_counter()
    state = graph.invoke({"run_id": run_id, "question": req.question, "period": req.period, "compare_to": req.compare_to})
    answered = state["answer_type"] in ("decomposition", "recommendation", "calculated")
    res = state.get("results", {}) if answered else {}
    actions = res.get("estimate_savings", {}).get("actions", []) if state["answer_type"] != "not_in_data" else []
    data = {k: res[k] for k in ("asset_breakdown", "find_idle_load", "what_if_tariff", "what_if_actions", "query_validated_data", "bill_total",
                                "detect_new_waste", "find_off_hours_waste", "check_asset", "carbon_footprint", "compare_data_versions") if k in res}
    if "decompose_change" in res:
        b = res["decompose_change"]
        data.update({"total_change_pct": b["total_change_pct"], "bill_reference": b["bill_reference"], "bill_current": b["bill_current"],
                     "baseline": b["baseline"], "waste_confirmed": b["waste_confirmed"]})
    data.update({"period": state.get("period"), "compare_to": state.get("compare_to"), "currency": ws.currency()})
    response = {"run_id": run_id, "answer_type": state["answer_type"], "summary": state["summary"],
                "bridge": res.get("decompose_change", {}).get("bridge", []),
                "recommendations": [_recommendation(a) for a in actions], "data": data,
                "assumptions": list(dict.fromkeys(a for a in state.get("assumptions", []) if a)) if answered else [],
                "limitations": list(dict.fromkeys(x for x in state.get("limitations", []) if x))}
    ws.save_trace({"run_id": run_id, "question": req.question, "period": state.get("period"),
                   "compare_to": state.get("compare_to"), "answer_type": state["answer_type"],
                   "steps": state.get("steps", []), "llm_calls": state.get("llm_calls", 0),
                   "tokens": state.get("tokens", 0), "llm_errors": state.get("llm_errors", []),
                   "llm_provider": llm.provider(), "llm_model": llm.model(), "data_version": ws.version,
                   "latency_ms": round(1000 * (time.perf_counter() - t0)),
                   "approval_status": "pending" if actions else "not_applicable", "response": response})
    return response


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), table: str | None = Form(default=None), meta: str | None = Form(default=None),
                 mode: str = Form(default="auto")) -> dict:
    """Multipart: a CSV file plus either a `table` field or a `meta` JSON such as {"table": "tariff"}.
    Optional `mode`: auto (default) or replace."""
    if meta:
        try:
            m = json.loads(meta)
            table, mode = table or m.get("table"), m.get("mode", mode)
        except (ValueError, AttributeError):
            raise HTTPException(400, "meta must be JSON such as {\"table\": \"tariff\"}")
    if table not in SCHEMA:
        raise HTTPException(400, f"table must be one of {list(SCHEMA)}")
    if mode not in ("auto", "replace"):
        raise HTTPException(400, "mode must be auto or replace")
    try:
        raw = pd.read_csv(io.BytesIO(await file.read()))
        out = ws.ingest(table, raw, mode)
    except Exception as e:  # unreadable file or missing columns: report, never crash
        return {"rows_loaded": 0, "rows_rejected": 0, "status": "rejected", "table": table, "detail": str(e)[:300]}
    out["table"] = table
    if ws.ready:
        out["validation"] = ws.checks()
    return out


@app.get("/health")
def health() -> dict:
    tables = {name: int(len(ws.frames[name])) if name in ws.frames else 0 for name in SCHEMA}
    return {"api": "ok", "data_store": {"status": "ready" if ws.ready else "waiting for data", "engine": "duckdb", "rows": tables,
                                         "quarters": ws.quarters() if ws.ready else [], "version": ws.version},
            "llm": llm.status()}


@app.get("/validation")
def validation() -> dict:
    if not ws.ready:
        raise HTTPException(409, "load all four tables first")
    return {"checks": ws.checks(), "cleaning": ws.cleaned()[1], "ingest_log": ws.ingest_log()}


@app.get("/trace/{run_id}")
def trace(run_id: str) -> dict:
    record = ws.get_trace(run_id)
    if record is None:
        raise HTTPException(404, f"no run {run_id}")
    return record


# ---- read-only dashboard data (the UI computes nothing itself) ------------------------------

def _periods(period: str | None, compare_to: str | None) -> tuple[str, str]:
    if not ws.ready:
        raise HTTPException(409, "load all four tables first")
    quarters = ws.quarters()
    p0, _ = ws.default_periods()
    period = period or p0
    if period not in quarters:
        raise HTTPException(404, f"{period} is not in the data ({', '.join(quarters)})")
    compare_to = compare_to or (quarters[quarters.index(period) - 1] if quarters.index(period) > 0 else None)
    if compare_to not in quarters:
        raise HTTPException(404, "need an earlier quarter to compare with")
    return period, compare_to


@app.get("/dashboard")
def dashboard(period: str | None = None, compare_to: str | None = None) -> dict:
    period, compare_to = _periods(period, compare_to)
    meters, _ = ws.cleaned()
    b = ws.bridge(period, compare_to)
    actions = ws.actions(period, compare_to)
    checks = ws.checks()
    factor, source = ws.emission_factor()
    return {"period": period, "compare_to": compare_to, "quarters": ws.quarters(), "currency": ws.currency(),
            "emission_factor": factor, "emission_factor_source": source, "data_version": ws.version,
            "bridge": b, "actions": [_recommendation(a) | {"window": a["window"]} for a in actions],
            "savings": an.savings_summary(actions), "new_waste": ws.leaks(period, compare_to),
            "off_hours_waste": ws.findings(period, compare_to), "assets": an.asset_breakdown(meters, period, compare_to),
            "idle": an.idle_load(meters, ws.frames, period), "daily": an.daily_series(ws.daily(compare_to)),
            "tariff_changes": sorted({str(t.date()) for t in ws.frames["tariff"]["valid_from"]})[1:],
            "changes": ws.changes(), "checks": {"passed": sum(c["passed"] for c in checks), "total": len(checks),
                                                "flags": [c for c in checks if not c["passed"]]},
            "llm": llm.status(ping=False),
            "late_drop_available": bool(ws._meta("last_folder")) and (Path(ws._meta("last_folder")) / "late_drop").is_dir()}


@app.get("/profile/{asset}")
def profile(asset: str, period: str | None = None) -> dict:
    period, _ = _periods(period, None)
    meters, _ = ws.cleaned()
    if asset not in meters.columns or asset == "main":
        raise HTTPException(404, f"no sub-meter {asset}")
    return an.load_profile(meters, ws.frames, period, asset)


@app.post("/actions/{action_key}/decision")
def decide(action_key: str, body: Decision) -> dict:
    return ws.approve(action_key, body.decision, body.note)


# ---- admin -----------------------------------------------------------------------------

class LoadRequest(BaseModel):
    folder: str | None = None
    sample_seed: int | None = None
    late_drop: bool = False


def _apply_late_drop(folder: Path) -> dict:
    report = {}
    for name in SCHEMA:
        f = folder / "late_drop" / f"{name}.csv"
        if f.exists():
            report[f"late_drop/{name}"] = ws.ingest(name, pd.read_csv(f))
    return report


@app.post("/admin/load")
def admin_load(body: LoadRequest) -> dict:
    """Replace the data with a folder of CSVs on the server (the loader for judges' data), or with a
    generated synthetic sample. With late_drop, the folder's late_drop files are applied after."""
    if body.sample_seed is not None:
        from .generate import generate

        folder = ws.dir / "samples" / f"seed_{body.sample_seed}"
        if not (folder / "meter_15min.csv").exists():
            generate(body.sample_seed, folder)
    elif body.folder:
        folder = Path(body.folder)
        if not folder.is_dir():
            raise HTTPException(404, f"no folder {body.folder} on the server")
    else:
        raise HTTPException(400, "give folder or sample_seed")
    report = ws.load_folder(folder)
    if body.late_drop:
        report.update(_apply_late_drop(folder))
    return {"loaded": report, "ready": ws.ready, "validation": ws.checks() if ws.ready else []}


@app.post("/admin/late-drop")
def admin_late_drop() -> dict:
    """Apply the late_drop files that sit next to the last loaded folder (the demo's late data drop)."""
    folder = ws._meta("last_folder")
    if not folder or not (Path(folder) / "late_drop").is_dir():
        raise HTTPException(404, "the last loaded folder has no late_drop files; upload them through /ingest instead")
    report = _apply_late_drop(Path(folder))
    return {"loaded": report, "changes": ws.changes()}


@app.post("/admin/reset")
def admin_reset() -> dict:
    ws.reset()
    return {"status": "empty"}
