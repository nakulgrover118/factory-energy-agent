"""LangGraph orchestration: route -> tools (loop, capped) -> verify -> compose -> number check.

Only `route` (as a fallback) and `compose` may call an LLM. Every number comes from the
tools in analytics.py. Verification gates the wording, and the number check gates the answer.
"""
from __future__ import annotations

import math
import re
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import analytics as an
from . import llm
from .store import Workspace

MAX_LOOPS = 4
CAUSE_LABEL = {"tariff": "tariff", "volume": "volume", "product_mix": "product mix", "weather": "weather", "waste": "waste"}


class State(TypedDict, total=False):
    run_id: str
    question: str
    period: str | None
    compare_to: str | None
    intent: str
    params: dict
    plan: list[str]
    results: dict
    loops: int
    replans: int
    verified: bool
    verify_note: str
    answer_type: str
    summary: str
    facts: dict
    steps: list[dict]
    llm_calls: int
    tokens: int
    assumptions: list[str]
    limitations: list[str]


PLANS = {"bridge": ["decompose", "detect_leaks", "estimate_savings"], "top_users": ["asset_breakdown"],
         "idle": ["idle_load"], "recommend": ["detect_leaks", "estimate_savings"],
         "what_if": ["what_if_tariff"], "meter_query": ["query_meter"]}
ANSWER_TYPE = {"bridge": "decomposition", "top_users": "calculated", "idle": "calculated", "recommend": "recommendation",
               "what_if": "calculated", "meter_query": "calculated"}

REFUSE = r"\b(make up|invent|fabricat|pretend|guess a|just say|ignore (the )?(data|rules|validation|checks)|skip (the )?(validation|checks)|without (checking|the data)|approve (it|this|the action)|auto-?approve)"
OUT_OF_SCOPE = r"\b(next (year|quarter|month)|forecast|predict|will the (tariff|price|bill)|going to be|in 20(2[7-9]|[3-9]\d))"
WHAT_IF = r"\b(what if|what would|suppose|if the (tariff|price)|had not (changed|risen|gone up)|hadn.t (changed|risen))"
IDLE = r"\b(idle|nothing is (being )?produced|no production|not producing|when .* (closed|shut)|standby|non-?working)"
RECOMMEND = r"\b(what should we do|do first|recommend|action|pay(s)? back|payback|saving|fix|leak|waste|reduce)"
BRIDGE = r"\b(why|bill|cost|increase|rise|rose|went up|go up|gone up|decompos|bridge|breakdown of the (bill|increase)|cause)"
TOP = r"\b(which (asset|meter|machine|equipment)s?|most energy|biggest|largest|top (user|consumer)|rank)"
ENERGY = r"\b(how much|energy|kwh|consum|use[ds]?)\b"


def _meter_in(question: str, meters: list[str]) -> str | None:
    q = question.lower().replace("-", " ").replace("_", " ")
    for m in sorted(meters, key=len, reverse=True):
        name = m.replace("_", " ")
        if name in q or name.rstrip("s") in q:
            return m
    return None


def keyword_route(question: str, meters: list[str]) -> tuple[str, dict]:
    q = question.lower()
    has = lambda pattern: re.search(pattern, q) is not None
    if has(REFUSE):
        return "refuse", {}
    if has(OUT_OF_SCOPE):
        return "out_of_scope", {}
    if has(WHAT_IF):
        if has(r"not (changed|risen|gone up)|n.t (changed|risen)|no (tariff )?(change|rise|increase)|stayed"):
            return "what_if", {"factor": 0.0}
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(lower|smaller|less)", q) or re.search(r"half", q)
        if m:
            return "what_if", {"factor": 0.5 if m.group(0) == "half" else max(0.0, 1 - float(m.group(1)) / 100)}
        return "unclear", {"ask": "Which tariff scenario: no tariff change, or the rise smaller by what percentage?"}
    if has(IDLE):
        return "idle", {}
    if has(TOP):
        return "top_users", {}
    if has(RECOMMEND):
        return "recommend", {}
    meter = _meter_in(question, meters)
    if meter and has(ENERGY) and not has(r"\bwhy\b"):
        return "meter_query", {"meter": meter}
    if has(BRIDGE):
        return "bridge", {}
    return "unknown", {}


def _numbers(obj: Any) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)] if math.isfinite(obj) else []
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _numbers(v)]
    if isinstance(obj, (list, tuple)):
        return [n for v in obj for n in _numbers(v)]
    if isinstance(obj, str):
        return _text_numbers(obj)
    return []


def _text_numbers(text: str) -> list[float]:
    text = re.sub(r"\d{4}-\d{2}-\d{2}|\d{4}-Q\d|\bQ\d\b|\d{1,2}:\d{2}|\b\d{2}-\d{2}h\b", " ", text)  # dates, quarters, clock times
    return [float(m.replace(",", "")) for m in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", text)]


def unmatched_numbers(text: str, facts: Any) -> list[str]:
    """Numbers in the text that no tool produced. A text number matches a fact when the fact,
    rounded to the precision shown in the text, equals it (sign ignored)."""
    allowed = [abs(n) for n in _numbers(facts)]
    cleaned = re.sub(r"\d{4}-\d{2}-\d{2}|\d{4}-Q\d|\bQ\d\b|\d{1,2}:\d{2}|\b\d{2}-\d{2}h\b", " ", text)
    bad = []
    for token in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", cleaned):
        raw = token.replace(",", "")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        value = float(raw)
        if not any(abs(round(a, decimals) - value) < 1e-9 for a in allowed):
            bad.append(token)
    return bad


def build_graph(ws: Workspace):
    def step(state: State, agent: str, tool: str, inp: Any = None, out: Any = None) -> list[dict]:
        entry = {"agent": agent, "tool": tool}
        if inp is not None:
            entry["input"] = inp
        if out is not None:
            entry["output"] = out
        return state.get("steps", []) + [entry]

    def route(state: State) -> dict:
        upd: dict = {"results": {}, "loops": 0, "replans": 0, "llm_calls": 0, "tokens": 0,
                     "assumptions": [], "limitations": []}
        if not ws.ready:
            return {**upd, "intent": "out_of_scope", "plan": [], "params": {},
                    "summary": "No complete data set is loaded. Load meter_15min, tariff, production and weather first.",
                    "steps": step(state, "orchestrator", "route", out={"intent": "no_data"})}
        meters = [c for c in ws.cleaned()[0].columns if c != "main"]
        intent, params = keyword_route(state["question"], meters)
        used = "keywords"
        if intent == "unknown":
            label, tokens = llm.classify(state["question"])
            if tokens:
                upd["llm_calls"], upd["tokens"], used = 1, tokens, "llm"
            intent = label if label and label not in ("what_if", "meter_query") else "unclear"
        quarters = ws.quarters()
        period = state.get("period") or (quarters[-1] if quarters else None)
        compare = state.get("compare_to") or (quarters[quarters.index(period) - 1] if period in quarters and quarters.index(period) > 0 else None)
        if intent in PLANS and (period not in quarters or (intent in ("bridge", "what_if", "recommend") and compare not in quarters)):
            intent = "out_of_scope"
            upd["summary"] = (f"The loaded data covers {', '.join(quarters) or 'no full quarter'}; "
                              f"it cannot answer for {period} compared with {compare}.")
        upd.update({"intent": intent, "params": params, "plan": list(PLANS.get(intent, [])),
                    "period": period, "compare_to": compare,
                    "steps": step(state, "orchestrator", "route", {"router": used},
                                  {"intent": intent, "plan": PLANS.get(intent, []), "period": period, "compare_to": compare})})
        return upd

    def run_tool(state: State) -> dict:
        name = state["plan"][0]
        period, compare, res = state["period"], state["compare_to"], dict(state["results"])
        meters, _ = ws.cleaned()
        inp: dict = {"period": period}
        if name == "decompose":
            inp["compare_to"] = compare
            out = ws.bridge(period, compare)
            brief = {"total_change_pct": out.get("total_change_pct"), "r2": out.get("baseline", {}).get("r2"),
                     **{b["cause"]: b["pct_points"] for b in out.get("bridge", [])}} if out.get("status") == "ok" else out
        elif name == "detect_leaks":
            inp["compare_to"] = compare
            out = an.detect_leaks(meters, ws.frames, period, compare)
            brief = [{"asset": f["asset"], "from": f["from"], "excess_kw": f["excess_kw"]} for f in out]
        elif name == "estimate_savings":
            factor, source = ws.emission_factor()
            inp["emission_factor"] = factor
            out = {"actions": an.estimate_savings(res.get("detect_leaks", []), meters, ws.frames, factor),
                   "emission_factor_source": source}
            brief = [{"action": a["action"], "payback_months": a["payback_months"]} for a in out["actions"]]
        elif name == "asset_breakdown":
            out = an.asset_breakdown(meters, period)
            brief = out[:3]
        elif name == "idle_load":
            out = an.idle_load(meters, ws.frames, period)
            brief = out[:3]
        elif name == "what_if_tariff":
            inp.update({"compare_to": compare, "price_rise_factor": state["params"]["factor"]})
            out = an.what_if_tariff(ws.daily(compare), period, compare, state["params"]["factor"])
            brief = out
        else:  # query_meter
            meter = state["params"]["meter"]
            inp["meter"] = meter
            q = meters[an.quarter_mask(meters.index, period)]
            out = {"meter": meter, "kwh": round(float(q[meter].sum()), 1),
                   "share_of_main": round(float(q[meter].sum() / q["main"].sum()), 4)}
            brief = out
        res[name] = out
        return {"results": res, "plan": state["plan"][1:], "loops": state["loops"] + 1,
                "steps": step(state, "analyst", name, inp, brief)}

    def verify(state: State) -> dict:
        res, problems = state["results"], []
        if state["loops"] >= MAX_LOOPS and state["plan"]:
            problems.append(f"plan not finished after {MAX_LOOPS} tool calls")
        b = res.get("decompose")
        if b is not None:
            if b.get("status") != "ok":
                problems.append(f"decomposition unavailable: {b.get('detail', b.get('status'))}")
            else:
                gap = abs(sum(x["pct_points"] for x in b["bridge"]) - b["total_change_pct"])
                if gap > 0.011:
                    problems.append(f"causes do not add up (gap {gap:.2f} points)")
                if any(not (x["range"][0] <= x["pct_points"] <= x["range"][1]) for x in b["bridge"]):
                    problems.append("a point estimate lies outside its range")
        for a in res.get("estimate_savings", {}).get("actions", []):
            keys = ("saving_kwh_per_year", "saving_money_per_year", "saving_co2e_t_per_year", "payback_months")
            if any(a[k][0] > a[k][1] for k in keys) or not a["evidence_refs"] or a["requires_approval"] is not True:
                problems.append(f"action '{a['action']}' has an invalid range, no evidence or no approval flag")
        if any(not math.isfinite(n) for n in _numbers({k: v for k, v in res.items() if k != "decompose"})):
            problems.append("a tool returned a non-finite number")
        ok = not problems
        upd: dict = {"verified": ok, "verify_note": "; ".join(problems),
                     "steps": step(state, "checker", "verify", out={"ok": ok, "problems": problems})}
        if not ok and state["replans"] < 1:  # one re-plan: recompute from the data, then give up honestly
            ws._changed()
            upd.update({"replans": state["replans"] + 1, "plan": list(PLANS[state["intent"]]), "results": {}, "loops": 0})
        return upd

    def compose(state: State) -> dict:
        intent, res = state["intent"], state.get("results", {})
        period, compare = state.get("period"), state.get("compare_to")
        assumptions, limitations = list(state.get("assumptions", [])), list(state.get("limitations", []))
        upd: dict = {}
        if intent == "refuse":
            return {"answer_type": "refused", "facts": {}, "steps": step(state, "writer", "compose", out={"answer_type": "refused"}),
                    "summary": "This request asks for a number or an approval that the data and rules do not support. "
                               "Figures come only from the loaded data, and every action needs a manager's approval."}
        if intent == "out_of_scope":
            return {"answer_type": "not_in_data", "facts": {}, "steps": step(state, "writer", "compose", out={"answer_type": "not_in_data"}),
                    "summary": state.get("summary") or "That is not in the data. The loaded files cover past meter readings, "
                               "tariffs, production and weather; they contain no forecasts, so no figure is given."}
        if intent == "unclear":
            ask = state.get("params", {}).get("ask") or ("Please say what you need: why the bill changed, which assets use the "
                                                         "most energy, idle-time energy, what to fix first, or a tariff what-if.")
            return {"answer_type": "clarification_needed", "facts": {}, "summary": ask,
                    "steps": step(state, "writer", "compose", out={"answer_type": "clarification_needed"})}
        if not state.get("verified"):
            return {"answer_type": "not_in_data", "facts": {}, "limitations": limitations + [state.get("verify_note", "")],
                    "summary": "The figures could not be verified, so no number is reported. " + state.get("verify_note", ""),
                    "steps": step(state, "writer", "compose", out={"answer_type": "not_in_data", "reason": "verification failed"})}

        facts: dict = {"period": period, "compare_to": compare}
        actions = res.get("estimate_savings", {}).get("actions", [])
        leaks = res.get("detect_leaks", [])
        if "estimate_savings" in res:
            factor, source = ws.emission_factor()
            assumptions += [f"Emission factor {factor} kg CO2e per kWh ({source})",
                            "Repair cost is an assumed range; the loaded data has no cost table"]
        if intent == "bridge":
            b = res["decompose"]
            facts.update({"total_change_pct": b["total_change_pct"],
                          "bridge": [{"cause": x["cause"], "pct_points": x["pct_points"], "range": x["range"]} for x in b["bridge"]],
                          "leaks": [{"asset": f["asset"], "from": f["from"], "rise_pct": f["rise_pct"]} for f in leaks]})
            ordered = sorted((x for x in b["bridge"] if x["cause"] != "waste"), key=lambda x: -abs(x["pct_points"]))
            parts = ", ".join(f"{CAUSE_LABEL[x['cause']]} {x['pct_points']:.2f} points ({x['range'][0]:.2f} to {x['range'][1]:.2f})" for x in ordered)
            waste = next(x for x in b["bridge"] if x["cause"] == "waste")
            direction = "rose" if b["total_change_pct"] >= 0 else "fell"
            text = (f"The bill {direction} {abs(b['total_change_pct']):.2f}% from {compare} to {period}. Causes outside the factory's control "
                    f"come first: {parts}. What remains is {waste['pct_points']:.2f} points ({waste['range'][0]:.2f} to {waste['range'][1]:.2f}) "
                    f"of unexplained energy, the only part treated as possible waste.")
            if leaks:
                f = leaks[0]
                text += f" Evidence of controllable waste: night load on {f['asset']} rose {f['rise_pct']}% from {f['from']} and stayed up."
            else:
                text += " No sub-meter shows a sustained rise in night load, so no asset is named."
            assumptions += b["assumptions"]
            limitations += [f"Baseline fit R2 {b['baseline']['r2']} on {compare}; ranges cover fit error only, not a wrong model form"]
        elif intent == "recommend":
            facts["actions"] = [{k: a[k] for k in ("action", "saving_kwh_per_year", "saving_money_per_year",
                                                   "saving_co2e_t_per_year", "payback_months", "ease", "rationale")} for a in actions]
            if actions:
                a = actions[0]
                text = (f"First action: {a['action'].lower()}. Estimated saving {a['saving_kwh_per_year'][0]:,} to {a['saving_kwh_per_year'][1]:,} kWh, "
                        f"{a['saving_money_per_year'][0]:,} to {a['saving_money_per_year'][1]:,} in cost and {a['saving_co2e_t_per_year'][0]} to "
                        f"{a['saving_co2e_t_per_year'][1]} t CO2e a year; payback {a['payback_months'][0]} to {a['payback_months'][1]} months, ease {a['ease']}. "
                        f"{a['rationale']} This is a suggestion and needs a manager's approval.")
                limitations.append("Payback depends on the assumed repair cost")
            else:
                text = f"No controllable waste pattern was found in {period}, so no action is recommended. Tariff, volume, mix and weather are not actions the factory can fix this way."
        elif intent == "top_users":
            rows = res["asset_breakdown"]
            facts["assets"] = rows
            top = [r for r in rows if r["asset"] != "unmetered"][:3]
            text = (f"In {period} the biggest users were " + ", ".join(f"{r['asset']} ({100 * r['share_of_main']:.1f}% of the main meter)" for r in top) + ". ")
            unmetered = next(r for r in rows if r["asset"] == "unmetered")
            text += f"{100 * unmetered['share_of_main']:.1f}% of the main meter is not covered by any sub-meter; it is reported as unmetered, not as waste."
            facts["shares_pct"] = [round(100 * r["share_of_main"], 1) for r in rows]
        elif intent == "idle":
            rows = res["idle_load"]
            facts["idle"] = rows
            total = round(sum(r["idle_kwh"] for r in rows), 1)
            facts["idle_total_kwh"] = total
            days = rows[0]["idle_days"] if rows else 0
            text = (f"On the {days} non-working days in {period} the sub-meters used {total:,.1f} kWh. Largest: "
                    + ", ".join(f"{r['asset']} {r['idle_kwh']:,.1f} kWh ({r['idle_kwh_per_day']:,.1f} a day)" for r in rows[:3]) + ".")
            limitations.append("Idle means days with no production; hours within working days are not counted")
        elif intent == "what_if":
            w = res["what_if_tariff"]
            facts.update(w)
            facts["price_rise_kept_pct"] = round(100 * state["params"]["factor"], 1)
            text = (f"With {facts['price_rise_kept_pct']}% of the tariff rise applied, the bill change from {compare} to {period} would be "
                    f"{w['scenario_change_pct']:.2f}% instead of {w['base_change_pct']:.2f}%, a difference of {abs(w['bill_difference']):,.2f} on the bill.")
            assumptions.append("Energy use is unchanged in the scenario; only the price differs")
        else:
            m = res["query_meter"]
            facts.update(m)
            facts["share_pct"] = round(100 * m["share_of_main"], 1)
            text = f"The {m['meter']} meter used {m['kwh']:,.1f} kWh in {period}, {facts['share_pct']}% of the main meter."

        upd.update({"answer_type": ANSWER_TYPE[intent], "facts": facts, "summary": text,
                    "assumptions": assumptions, "limitations": limitations})
        used = "template"
        if intent in ("bridge", "recommend"):  # the manager-facing briefs; short numeric answers stay as computed
            reworded, tokens = llm.reword(text, facts)
            if tokens:
                upd["llm_calls"], upd["tokens"] = state.get("llm_calls", 0) + 1, state.get("tokens", 0) + tokens
            if reworded:
                upd["summary"], used = reworded, "llm"
                upd["facts"] = {**facts, "_template": text}
        upd["steps"] = step(state, "writer", "compose", {"writer": used}, {"answer_type": upd["answer_type"]})
        return upd

    def number_check(state: State) -> dict:
        facts = state.get("facts", {})
        bad = unmatched_numbers(state["summary"], {k: v for k, v in facts.items() if k != "_template"})
        upd: dict = {}
        if bad and "_template" in facts:  # the LLM wording carried a number no tool produced: use the template
            upd["summary"] = facts["_template"]
            upd["limitations"] = state.get("limitations", []) + ["LLM wording was rejected by the number check; template wording used"]
            bad_after = unmatched_numbers(upd["summary"], {k: v for k, v in facts.items() if k != "_template"})
        else:
            bad_after = bad
        if bad_after:
            upd.update({"answer_type": "not_in_data", "summary": "The answer was withheld because a figure in it could not be traced to a tool result."})
        upd["steps"] = step(state, "checker", "number_check", out={"unmatched_first_pass": bad, "unmatched_final": bad_after})
        return upd

    g = StateGraph(State)
    for name, fn in (("route", route), ("run_tool", run_tool), ("verify", verify), ("compose", compose), ("number_check", number_check)):
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda s: "run_tool" if s["plan"] else "compose", ["run_tool", "compose"])
    g.add_conditional_edges("run_tool", lambda s: "run_tool" if s["plan"] and s["loops"] < MAX_LOOPS else "verify", ["run_tool", "verify"])
    g.add_conditional_edges("verify", lambda s: "run_tool" if (not s["verified"] and s["plan"]) else "compose", ["run_tool", "compose"])
    g.add_edge("compose", "number_check")
    g.add_edge("number_check", END)
    return g.compile()


def new_run_id() -> str:
    return "en-" + uuid.uuid4().hex[:8]
