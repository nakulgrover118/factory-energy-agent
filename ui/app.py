"""Streamlit front end. It only calls the API; it computes nothing itself.

Run: streamlit run ui/app.py   (API_URL defaults to http://localhost:8000)
"""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TABLES = ["meter_15min", "tariff", "production", "weather"]
EXAMPLES = ["Why did the bill go up 15%?", "Which assets use the most energy?",
            "How much energy is used when nothing is produced?", "What should we do first?",
            "What if the tariff had not changed?", "What will the tariff be next year?"]
CAUSE_LABEL = {"tariff": "Tariff", "volume": "Volume", "product_mix": "Product mix", "weather": "Weather", "waste": "Waste"}
TYPE_NOTE = {"decomposition": "Split of the bill change", "recommendation": "Suggested actions",
             "calculated": "Calculated from the data", "not_in_data": "Not in the data",
             "clarification_needed": "More detail needed", "refused": "Request declined"}

st.set_page_config(page_title="Factory energy bill explainer", layout="wide")


def call(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API}{path}", timeout=120, **kwargs)
        return (r.json(), None) if r.ok else (None, f"{r.status_code}: {r.text[:300]}")
    except requests.RequestException as e:
        return None, f"Cannot reach the API at {API} ({type(e).__name__})"


def bridge_chart(bridge: list[dict], total: float, period: str, compare_to: str) -> go.Figure:
    order = sorted((b for b in bridge if b["cause"] != "waste"), key=lambda b: -abs(b["pct_points"]))
    order += [b for b in bridge if b["cause"] == "waste"]
    labels = [f"{compare_to} bill"] + [CAUSE_LABEL.get(b["cause"], b["cause"]) for b in order] + [f"{period} bill"]
    values = [100.0] + [b["pct_points"] for b in order] + [0.0]
    text = ["100"] + [f"{b['pct_points']:+.2f}<br>({b['range'][0]:.2f} to {b['range'][1]:.2f})" for b in order] + [f"{100 + total:.2f}"]
    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=["absolute"] + ["relative"] * len(order) + ["total"], text=text, textposition="outside",
        connector={"line": {"color": "#9AA5B1", "width": 1}},
        increasing={"marker": {"color": "#0B6E6E"}}, decreasing={"marker": {"color": "#B4440A"}},
        totals={"marker": {"color": "#3F4F5E"}}))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      yaxis=dict(title=f"Index, {compare_to} bill = 100", range=[90, 100 + max(total, 0) + 8]))
    return fig


# ---- sidebar: status, data loading, validation --------------------------------
with st.sidebar:
    st.header("Data and status")
    health, err = call("GET", "/health")
    if err:
        st.error(err)
        st.stop()
    store = health["data_store"]
    st.write(f"**Data:** {store['status']}" + (f" ({', '.join(store['quarters'])})" if store["quarters"] else ""))
    st.write(f"**LLM:** {health['llm']['provider']} ({health['llm']['status']})")
    st.caption("Rows loaded: " + ", ".join(f"{k} {v:,}" for k, v in store["rows"].items()))

    with st.expander("Load a CSV file", expanded=store["status"] != "ready"):
        table = st.selectbox("Table", TABLES)
        upload = st.file_uploader("CSV in the documented schema", type="csv")
        if st.button("Load", disabled=upload is None):
            out, err = call("POST", "/ingest", files={"file": (upload.name, upload.getvalue(), "text/csv")},
                            data={"meta": json.dumps({"table": table})})
            if err:
                st.error(err)
            elif out["status"] == "loaded":
                st.success(f"Loaded {out['rows_loaded']:,} rows, rejected {out['rows_rejected']:,}. Results are recomputed.")
            else:
                st.error(f"Rejected: {out.get('detail', 'no valid rows')}")

    if store["status"] == "ready":
        val, err = call("GET", "/validation")
        if val:
            st.subheader("Validation")
            for c in val["checks"]:
                (st.success if c["passed"] else st.warning)(f"{c['id']}. {c['check']}: {c['detail']}")
            with st.expander("Cleaning report"):
                st.json(val["cleaning"])

# ---- main: ask and read the answer --------------------------------------------
st.title("Factory energy bill explainer")
st.caption("Every number below is computed by code from the loaded data. Actions are suggestions for a manager to approve.")

if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0]
cols = st.columns(3)
for i, example in enumerate(EXAMPLES):
    if cols[i % 3].button(example, use_container_width=True):
        st.session_state.question = example

with st.form("ask"):
    question = st.text_input("Question", key="question")
    c1, c2, _ = st.columns([1, 1, 3])
    period = c1.text_input("Period (optional)", placeholder="2026-Q3")
    compare_to = c2.text_input("Compare to (optional)", placeholder="2026-Q2")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and question.strip():
    body = {"question": question.strip()}
    if period.strip():
        body["period"] = period.strip()
    if compare_to.strip():
        body["compare_to"] = compare_to.strip()
    with st.spinner("Working through the data"):
        st.session_state.answer, st.session_state.error = call("POST", "/analyze", json=body)

if st.session_state.get("error"):
    st.error(st.session_state.error)
answer = st.session_state.get("answer")
if answer:
    kind = answer["answer_type"]
    st.subheader(TYPE_NOTE.get(kind, kind))
    box = st.info if kind in ("decomposition", "recommendation", "calculated") else st.warning
    box(answer["summary"])
    data = answer.get("data", {})

    if answer["bridge"]:
        trace, _ = call("GET", f"/trace/{answer['run_id']}")
        p, c = (trace or {}).get("period", "current"), (trace or {}).get("compare_to", "reference")
        st.plotly_chart(bridge_chart(answer["bridge"], data["total_change_pct"], p, c), use_container_width=True)
        table = pd.DataFrame([{"Cause": CAUSE_LABEL.get(b["cause"], b["cause"]), "Points": b["pct_points"],
                               "Low": b["range"][0], "High": b["range"][1]} for b in answer["bridge"]])
        left, right = st.columns([2, 1])
        left.dataframe(table, hide_index=True, use_container_width=True)
        right.metric("Bill change", f"{data['total_change_pct']:.2f}%")
        right.metric("Baseline fit (R2)", f"{data['baseline']['r2']:.3f}")

    if answer["recommendations"]:
        st.subheader("Suggested actions (need approval)")
        rng = lambda v: f"{v[0]:,} to {v[1]:,}"
        st.dataframe(pd.DataFrame([{"Action": r["action"], "kWh a year": rng(r["saving_kwh_per_year"]),
                                    "Money a year": rng(r["saving_money_per_year"]),
                                    "CO2e t a year": rng(r["saving_co2e_t_per_year"]),
                                    "Payback months": rng(r["payback_months"]), "Ease": r["ease"],
                                    "Confidence": r["confidence"]} for r in answer["recommendations"]]),
                     hide_index=True, use_container_width=True)
        for r in answer["recommendations"]:
            st.caption(f"{r['action']}: {r['rationale']} Evidence: {', '.join(r['evidence_refs'])}")

    for key, title in (("asset_breakdown", "Energy by asset"), ("idle_load", "Energy on non-working days"),
                       ("detect_leaks", "Waste findings")):
        if data.get(key):
            st.subheader(title)
            st.dataframe(pd.DataFrame(data[key]), hide_index=True, use_container_width=True)
    for key in ("what_if_tariff", "query_meter"):
        if data.get(key):
            st.json(data[key])

    left, right = st.columns(2)
    if answer["assumptions"]:
        left.markdown("**Assumptions**\n" + "\n".join(f"- {a}" for a in answer["assumptions"]))
    if answer["limitations"]:
        right.markdown("**Limitations**\n" + "\n".join(f"- {a}" for a in answer["limitations"]))

    with st.expander(f"Trace for run {answer['run_id']}"):
        trace, err = call("GET", f"/trace/{answer['run_id']}")
        if trace:
            st.dataframe(pd.DataFrame([{"Step": i + 1, "Agent": s["agent"], "Tool": s["tool"],
                                        "Input": json.dumps(s.get("input", "")), "Output": json.dumps(s.get("output", ""))[:300]}
                                       for i, s in enumerate(trace["steps"])]), hide_index=True, use_container_width=True)
            st.caption(f"LLM calls {trace['llm_calls']}, tokens {trace['tokens']}, {trace['latency_ms']} ms, provider {trace['llm_provider']}")
        else:
            st.error(err)
