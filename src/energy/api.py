"""FastAPI contract: POST /analyze, POST /ingest, GET /health, GET /trace/{run_id}.

Run: uvicorn energy.api:app --app-dir src --port 8000
"""
from __future__ import annotations

import io
import json
import time
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import llm
from .agent import build_graph, new_run_id
from .pipeline import SCHEMA
from .store import Workspace

app = FastAPI(title="Factory Energy Cost Decomposition and Savings Agent", version="0.1.0")
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
    action: str
    rationale: str
    evidence_refs: list[str]
    confidence: str
    estimated_impact: str
    requires_approval: bool
    saving_kwh_per_year: list[float]
    saving_money_per_year: list[float]
    saving_co2e_t_per_year: list[float]
    payback_months: list[float]
    ease: str


class AnalyzeResponse(BaseModel):
    run_id: str
    answer_type: str
    summary: str
    bridge: list[BridgeItem] = []
    recommendations: list[Recommendation] = []
    data: dict[str, Any] = {}
    assumptions: list[str] = []
    limitations: list[str] = []


def _recommendation(a: dict) -> dict:
    return {**{k: a[k] for k in ("action", "rationale", "evidence_refs", "confidence", "requires_approval", "ease",
                                 "saving_kwh_per_year", "saving_money_per_year", "saving_co2e_t_per_year", "payback_months")},
            "estimated_impact": (f"Saves {a['saving_kwh_per_year'][0]:,} to {a['saving_kwh_per_year'][1]:,} kWh and "
                                 f"{a['saving_money_per_year'][0]:,} to {a['saving_money_per_year'][1]:,} a year")}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> dict:
    run_id, t0 = new_run_id(), time.perf_counter()
    state = graph.invoke({"run_id": run_id, "question": req.question, "period": req.period, "compare_to": req.compare_to})
    answered = state["answer_type"] in ("decomposition", "recommendation", "calculated")
    res = state.get("results", {}) if answered else {}
    actions = res.get("estimate_savings", {}).get("actions", [])
    data = {k: res[k] for k in ("asset_breakdown", "idle_load", "what_if_tariff", "query_meter", "detect_leaks") if k in res}
    if "decompose" in res:
        b = res["decompose"]
        data.update({"total_change_pct": b["total_change_pct"], "bill_reference": b["bill_reference"],
                     "bill_current": b["bill_current"], "baseline": b["baseline"]})
    response = {"run_id": run_id, "answer_type": state["answer_type"], "summary": state["summary"],
                "bridge": res.get("decompose", {}).get("bridge", []),
                "recommendations": [_recommendation(a) for a in actions], "data": data,
                "assumptions": [a for a in state.get("assumptions", []) if a] if answered else [],
                "limitations": [x for x in state.get("limitations", []) if x]}
    ws.save_trace({"run_id": run_id, "question": req.question, "period": state.get("period"),
                   "compare_to": state.get("compare_to"), "answer_type": state["answer_type"],
                   "steps": state.get("steps", []), "llm_calls": state.get("llm_calls", 0),
                   "tokens": state.get("tokens", 0), "llm_provider": llm.provider(), "data_version": ws.version,
                   "latency_ms": round(1000 * (time.perf_counter() - t0)),
                   "approval_status": "pending" if actions else "not_applicable", "response": response})
    return response


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), table: str | None = Form(default=None), meta: str | None = Form(default=None)) -> dict:
    """Multipart: a CSV file plus either a `table` field or a `meta` JSON such as {"table": "tariff"}."""
    if not table and meta:
        try:
            table = json.loads(meta).get("table")
        except (ValueError, AttributeError):
            raise HTTPException(400, "meta must be JSON such as {\"table\": \"tariff\"}")
    if table not in SCHEMA:
        raise HTTPException(400, f"table must be one of {list(SCHEMA)}")
    try:
        raw = pd.read_csv(io.BytesIO(await file.read()))
        out = ws.ingest(table, raw)
    except Exception as e:  # unreadable file or missing columns: report, never crash
        return {"rows_loaded": 0, "rows_rejected": 0, "status": "rejected", "detail": str(e)[:300]}
    out["table"] = table
    if ws.ready:
        out["validation"] = ws.checks()
    return out


@app.get("/health")
def health() -> dict:
    tables = {name: int(len(ws.frames[name])) if name in ws.frames else 0 for name in SCHEMA}
    return {"api": "ok", "data_store": {"status": "ready" if ws.ready else "waiting for data", "rows": tables,
                                         "quarters": ws.quarters() if ws.ready else [], "version": ws.version},
            "llm": llm.status()}


@app.get("/validation")
def validation() -> dict:
    if not ws.ready:
        raise HTTPException(409, "load all four tables first")
    return {"checks": ws.checks(), "cleaning": ws.cleaned()[1]}


@app.get("/trace/{run_id}")
def trace(run_id: str) -> dict:
    record = ws.get_trace(run_id)
    if record is None:
        raise HTTPException(404, f"no run {run_id}")
    return record
