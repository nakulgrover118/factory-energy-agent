"""LangGraph orchestration: route -> tools (loop, capped) -> verify -> compose -> number check.

Only `route` (reading the question) and `compose` (rewording the brief) may call an LLM. Every
number comes from the tools in analytics.py and store.py. Verification gates the wording, and
the number check gates the answer.
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

MAX_LOOPS = 6
CAUSE_LABEL = {"tariff": "tariff", "volume": "volume", "product_mix": "product mix", "weather": "weather", "waste": "waste"}
CURRENCY_SYMBOL = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}
ASSET_NAME = {"hvac": "HVAC", "compressors": "compressors", "furnace": "furnace", "lighting": "lighting", "utilities": "utilities"}


def label(asset: str) -> str:
    return ASSET_NAME.get(asset, asset.replace("_", " ").replace("line", "line").strip())


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
    llm_errors: list[str]
    assumptions: list[str]
    limitations: list[str]


WASTE_TOOLS = ["detect_new_waste", "find_off_hours_waste", "estimate_savings"]
PLANS = {"bridge": ["decompose_change"] + WASTE_TOOLS,
         "asset_check": ["decompose_change", "detect_new_waste", "find_off_hours_waste", "check_asset"],
         "top_users": ["asset_breakdown"], "idle": ["find_idle_load", "detect_new_waste", "find_off_hours_waste"],
         "recommend": WASTE_TOOLS, "what_if_tariff": ["what_if_tariff"], "what_if_actions": WASTE_TOOLS + ["what_if_actions"],
         "meter_query": ["query_validated_data"], "footprint": ["carbon_footprint"], "changes": ["compare_data_versions"],
         "bill_total": ["bill_total"]}
ANSWER_TYPE = {"bridge": "decomposition", "asset_check": "decomposition", "top_users": "calculated", "idle": "calculated",
               "recommend": "recommendation", "what_if_tariff": "calculated", "what_if_actions": "calculated",
               "meter_query": "calculated", "footprint": "calculated", "changes": "decomposition", "bill_total": "calculated"}
NEEDS_COMPARE = {"bridge", "asset_check", "what_if_tariff", "what_if_actions", "recommend", "idle", "changes"}

REFUSE = r"\b(make up|invent|fabricat|pretend|guess (a|the|me)|just (say|give me) a number|ignore (the )?(data|rules|validation|checks)|skip (the )?(validation|checks)|without (checking|the data)|approve (it|this|the action|all)|auto-?approve)"
OUT_OF_SCOPE = r"\b(next (year|quarter|month)|forecast|predict|will the (tariff|price|bill)|going to be|in 20(2[7-9]|[3-9]\d)|other (site|plant|factor(y|ies)))"
CHANGES = r"\b(what changed|late (data )?drop|corrected (data|tariff|file)|since the (update|new data|correction)|recalibrat|after the (update|correction|new data))"
WHAT_IF = r"\b(what if|what would|suppose|would the bill|if we (fix|repair|did|do|implement|switch)|if the (tariff|price)|had not (changed|risen|gone up)|hadn.t (changed|risen))"
ACTION_WORDS = r"\b(fix|repair|leak|action|recommend|implement|did (all|these|them)|switch(ed)? off|schedul)"
CO2 = r"\b(co2|carbon|emission|footprint|greenhouse)"
BLAME = r"\b(blame|caus|because of|responsible|fault|old|replac|upgrad|new model|wasting|to blame|behind|reason)"
UPGRADE = r"\b(replac|upgrad|new model|buy a new)"
IDLE = r"\b(idle|nothing is (being )?produced|no production|not producing|when .* (closed|shut)|standby|non-?working|weekend|sunday|holiday|at night|overnight)"
RECOMMEND = r"\b(rank|what should we do|do first|recommend|action|pay(s)? back|payback|saving|fix|leak|waste|reduce|cut (the )?(bill|cost))"
TOP = r"\b(which (asset|meter|machine|equipment)s?|most energy|biggest|largest|top (user|consumer)|breakdown by asset|by asset|energy breakdown)"
BRIDGE = r"\b(why|bill|cost|increase|rise|rose|went up|go up|gone up|decompos|bridge|cause|contribut|explain)"
ENERGY = r"\b(how much|energy|kwh|consum|use[ds]?)\b"
BILL_TOTAL = r"\b((total|what was the|how much was the|size of the) (bill|cost|energy cost)|bill (amount|total))"
FOCUS = [("tariff", r"\b(tariff|price|rate)"), ("weather", r"\b(weather|hot|heat|temperature|cooling degree)"),
         ("product_mix", r"\b(mix|product mix)"), ("volume", r"\b(volume|more units|production|output)"), ("waste", r"\b(waste|leak)")]
ALIASES = {"lights": "lighting", "light": "lighting", "air compressor": "compressors", "compressor": "compressors",
           "compressed air": "compressors", "air conditioning": "hvac", "ac": "hvac", "cooling": "hvac", "chiller": "hvac", "oven": "furnace", "kiln": "furnace"}


def _meter_in(question: str, meters: list[str]) -> str | None:
    q = " " + re.sub(r"[^a-z0-9 ]", " ", question.lower().replace("_", " ").replace("-", " ")) + " "
    for m in sorted(meters, key=len, reverse=True):
        name = m.replace("_", " ")
        if f" {name} " in q or f" {name.rstrip('s')} " in q:
            return m
    for word, m in ALIASES.items():
        if f" {word} " in q and m in meters:
            return m
    return None


def _quarter_in(question: str, years: list[str]) -> str | None:
    m = re.search(r"\b(20\d\d)[-\s]?q([1-4])\b|\bq([1-4])(?:[-\s]*(20\d\d))?\b", question.lower())
    if not m:
        return None
    year = m.group(1) or m.group(4) or (years[-1] if years else None)
    q = m.group(2) or m.group(3)
    return f"{year}-Q{q}" if year else None


def keyword_route(question: str, meters: list[str], quarters: list[str] | None = None) -> tuple[str, dict]:
    """Deterministic router: used in mock mode and whenever the LLM is unavailable."""
    q = question.lower()
    has = lambda pattern: re.search(pattern, q) is not None
    params: dict = {}
    years = sorted({x[:4] for x in quarters or []})
    found = [f"{y or (years[-1] if years else '')}-Q{n}" for y, n in
             ((m.group(1) or m.group(4), m.group(2) or m.group(3)) for m in
              re.finditer(r"\b(20\d\d)[-\s]?q([1-4])\b|\bq([1-4])(?:[-\s]*(20\d\d))?\b", q))]
    if len(found) >= 2:
        params["period"], params["compare_to"] = sorted(found[:2])[1], sorted(found[:2])[0]
    elif found:
        params["period"] = found[0]
    asset = _meter_in(question, meters)
    if has(REFUSE):
        return "refuse", params
    if has(OUT_OF_SCOPE):
        return "out_of_scope", params
    if has(CHANGES):
        return "changes", params
    if has(WHAT_IF):
        if has(ACTION_WORDS):
            return "what_if_actions", {**params, **({"asset": asset} if asset else {})}
        if has(r"\b(volume|production|output|units|produce)"):
            return "out_of_scope", {**params, "note": "Scenarios for a different production volume are not modelled; "
                                                     "the tool re-runs tariff and savings scenarios only, so no figure is given."}
        if has(r"not (changed|risen|gone up)|n.t (changed|risen)|no (tariff )?(change|rise|increase)|stayed|old (tariff|price)"):
            return "what_if_tariff", {**params, "rise_kept": 0.0}
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(lower|cheaper|less|smaller|down)", q)
        if m:
            return "what_if_tariff", {**params, "price_factor": round(1 - float(m.group(1)) / 100, 4)}
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(higher|more|up|dearer)", q)
        if m:
            return "what_if_tariff", {**params, "price_factor": round(1 + float(m.group(1)) / 100, 4)}
        if has(r"\bhalf\b"):
            return "what_if_tariff", {**params, "rise_kept": 0.5}
        return "unclear", {**params, "ask": "Which scenario: the tariff unchanged, prices a given percentage lower or higher, "
                                            "or the recommended actions done?"}
    if has(CO2):
        return "footprint", params
    if has(BILL_TOTAL):
        return "bill_total", params
    if asset and has(BLAME):
        return "asset_check", {**params, "asset": asset, "asset_request": "replace_or_upgrade" if has(UPGRADE) else "blame"}
    if has(IDLE) and not has(r"\b(rank|payback|pays back|do first)"):
        return "idle", {**params, **({"asset": asset} if asset else {})}
    if has(RECOMMEND):
        return "recommend", {**params, **({"asset": asset} if asset else {})}
    if has(TOP):
        return "top_users", params
    if asset and has(ENERGY) and not has(r"\bwhy\b"):
        return "meter_query", {**params, "asset": asset}
    if has(BRIDGE) or "compare_to" in params:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(%|per ?cent)", q)
        focus = next((c for c, pat in FOCUS if has(pat)), None) if has(r"\b(how much|contribut|due to|because of|share|part)") else None
        return "bridge", {**params, **({"stated_change_pct": float(m.group(1))} if m else {}), **({"focus_cause": focus} if focus else {})}
    return "unclear", params


# ---- number faithfulness -------------------------------------------------------------

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


_NOT_FIGURES = r"\d{4}-\d{2}-\d{2}|\d{4}-Q\d|\bQ\d\b|\d{1,2}:\d{2}|\b\d{2}-\d{2}h\b|line_\d|line \d"


def _text_numbers(text: str) -> list[float]:
    text = re.sub(_NOT_FIGURES, " ", text)
    return [float(m.replace(",", "")) for m in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", text)]


def unmatched_numbers(text: str, facts: Any) -> list[str]:
    """Numbers in the text that no tool produced. A text number matches a fact when the fact,
    rounded to the precision shown in the text, equals it (sign ignored)."""
    allowed = [abs(n) for n in _numbers(facts)]
    cleaned = re.sub(_NOT_FIGURES, " ", text)
    bad = []
    for token in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", cleaned):
        raw = token.replace(",", "").rstrip(".")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        value = float(raw)
        if not any(abs(round(a, decimals) - value) < 1e-9 for a in allowed):
            bad.append(token)
    return bad


# ---- formatting helpers (every figure they print is added to the facts) ------------------

class Fmt:
    def __init__(self, currency: str):
        self.sym = CURRENCY_SYMBOL.get(currency, currency + " ")
        self.lakh = currency == "INR"
        self.shown: list[float] = []

    def money(self, x: float) -> str:
        if self.lakh:
            v = round(x / 1e5, 1)
            self.shown.append(v)
            return f"{self.sym}{v:,.1f} lakh"
        v = round(x)
        self.shown.append(v)
        return f"{self.sym}{v:,}"

    def money_range(self, r: list[float]) -> str:
        if self.lakh:
            a, b = round(r[0] / 1e5, 1), round(r[1] / 1e5, 1)
            self.shown += [a, b]
            return f"{self.sym}{a:,.1f}-{b:,.1f} lakh"
        return f"{self.money(r[0])} to {self.money(r[1])}"

    def num(self, x: float, nd: int = 0) -> str:
        v = round(x, nd)
        self.shown.append(v)
        return f"{v:,.{nd}f}"


def _when(f: dict) -> str:
    return "around the clock" if f["when"] == "24/7" else f"during {f['when']}"


def build_graph(ws: Workspace):
    def step(state: State, agent: str, tool: str, inp: Any = None, out: Any = None, **extra) -> list[dict]:
        entry = {"agent": agent, "tool": tool, **extra}
        if inp is not None:
            entry["input"] = inp
        if out is not None:
            entry["output"] = out
        return state.get("steps", []) + [entry]

    # ---- route -------------------------------------------------------------------------
    def route(state: State) -> dict:
        upd: dict = {"results": {}, "loops": 0, "replans": 0, "llm_calls": 0, "tokens": 0, "llm_errors": [],
                     "assumptions": [], "limitations": [], "params": {}}
        if not ws.ready:
            return {**upd, "intent": "out_of_scope", "plan": [],
                    "summary": "No complete data set is loaded. Load meter_15min, tariff, production and weather first.",
                    "steps": step(state, "orchestrator", "route", out={"intent": "no_data"})}
        if (issue := ws.blocking_issue()):
            return {**upd, "intent": "out_of_scope", "plan": [], "summary": f"No figure is given: {issue}. Load the missing meter data first.",
                    "steps": step(state, "orchestrator", "route", out={"intent": "blocked", "reason": issue})}
        meters = [c for c in ws.cleaned()[0].columns if c != "main"]
        quarters = ws.quarters()
        question = state["question"]
        used, info = "keywords", {}
        if re.search(REFUSE, question.lower()):
            intent, params = "refuse", {}
        else:
            intent, params = None, {}
            if llm.provider() != "mock":
                r = llm.route(question, meters, quarters)
                upd["llm_calls"], upd["tokens"] = 1, r["tokens"]
                info = {"model": r["model"], "ms": r["ms"], "tokens": r["tokens"]}
                if r["error"]:
                    upd["llm_errors"].append(f"route: {r['error']}")
                    info["error"] = r["error"]
                req = r.get("request")
                if req and req.get("intent") in llm.INTENTS:
                    used, intent = "llm", req["intent"]
                    params = {k: v for k, v in req.items() if k != "intent" and v is not None}
                    if params.get("asset") not in meters:
                        params.pop("asset", None)
                    for k in ("period", "compare_to"):
                        if k in params and not re.fullmatch(r"\d{4}-Q[1-4]", str(params[k])):
                            params.pop(k)
                    if intent != "bridge":
                        params.pop("stated_change_pct", None)
                    if intent == "unclear" and params.get("clarification"):
                        params["ask"] = params.pop("clarification")
            if intent is None:
                intent, params = keyword_route(question, meters, quarters)
                used = "keywords" if not info else "keywords (LLM unavailable)"
            else:  # deterministic guards win over the model where the wording is unambiguous
                kw_intent, kw_params = keyword_route(question, meters, quarters)
                if ((kw_intent == "out_of_scope" and kw_params.get("note"))
                        or (intent == "refuse" and kw_intent == "asset_check")
                        or (intent == "what_if_tariff" and kw_intent != "what_if_tariff")  # no scenario stated: ask, don't guess
                        or (intent == "idle" and kw_intent == "recommend" and re.search(r"\bleak", question.lower()))):
                    intent, params, used = kw_intent, kw_params, "llm + guard"
                if intent in ("asset_check", "recommend", "what_if_actions", "idle", "meter_query") and not params.get("asset") \
                        and kw_params.get("asset"):
                    params["asset"] = kw_params["asset"]
                for k in ("period", "compare_to"):
                    if kw_params.get(k) and not params.get(k):
                        params[k] = kw_params[k]
        if intent == "asset_check" and not params.get("asset"):
            intent = "bridge"
        if intent == "meter_query" and not params.get("asset"):
            intent = "top_users"
        if intent == "what_if_tariff" and params.get("price_factor") is None and params.get("rise_kept") is None:
            intent, params = "unclear", {"ask": "Which tariff scenario: unchanged, or prices a given percentage lower or higher?"}

        d_period, _ = ws.default_periods()
        period = state.get("period") or params.get("period") or d_period
        idx = quarters.index(period) if period in quarters else -1
        compare = state.get("compare_to") or params.get("compare_to") or (quarters[idx - 1] if idx > 0 else None)
        asked = [x for x in (params.get("period"), params.get("compare_to")) if x]
        if any(x not in quarters for x in asked) and intent not in ("refuse",):
            intent = "out_of_scope"
            upd["summary"] = (f"The loaded data covers {', '.join(quarters)}, so there is nothing to say about "
                              f"{', '.join(x for x in asked if x not in quarters)}; no figure is given.")
        elif intent in PLANS and (period not in quarters or (intent in NEEDS_COMPARE and compare not in quarters)):
            missing = period if period not in quarters else compare
            upd["summary"] = (f"The loaded data covers {', '.join(quarters) or 'no full quarter'}, so there is nothing to say about "
                              f"{missing or 'an earlier quarter'}; no figure is given.")
            intent = "out_of_scope"
        if intent == "out_of_scope" and params.get("note"):
            upd["summary"] = params["note"]
        upd.update({"intent": intent, "params": params, "plan": list(PLANS.get(intent, [])), "period": period, "compare_to": compare,
                    "steps": step(state, "orchestrator", "route", {"router": used, **info},
                                  {"intent": intent, "params": params, "plan": PLANS.get(intent, []), "period": period, "compare_to": compare})})
        return upd

    # ---- tools ---------------------------------------------------------------------------
    def run_tool(state: State) -> dict:
        name = state["plan"][0]
        period, compare, res, params = state["period"], state["compare_to"], dict(state["results"]), state.get("params", {})
        meters, _ = ws.cleaned()
        inp: dict = {"period": period, "compare_to": compare}
        if name == "decompose_change":
            out = ws.bridge(period, compare)
            brief = ({"total_change_pct": out["total_change_pct"], "baseline_r2": out["baseline"]["r2"],
                      **{b["cause"]: b["pct_points"] for b in out["bridge"]}} if out.get("status") == "ok" else out)
        elif name == "detect_new_waste":
            out = ws.leaks(period, compare)
            brief = [{"asset": f["asset"], "from": f["from"], "excess_kw": f["excess_kw"]} for f in out]
        elif name == "find_off_hours_waste":
            out = ws.findings(period, compare)
            brief = [{"asset": f["asset"], "window": f["window"], "excess_kw": f["excess_kw"]} for f in out]
        elif name == "estimate_savings":
            factor, source = ws.emission_factor()
            inp["emission_factor"] = factor
            out = {"actions": ws.actions(period, compare), "emission_factor_source": source}
            brief = [{"rank": a["rank"], "action": a["action"], "payback_months": a["payback_months"],
                      "saving_money_per_year": a["saving_money_per_year"]} for a in out["actions"]]
        elif name == "check_asset":
            inp["asset"] = params["asset"]
            out = an.asset_check(meters, ws.frames, params["asset"], period, compare, res.get("find_off_hours_waste", []),
                                 res.get("detect_new_waste", []))
            brief = {k: out[k] for k in ("asset", "change_pct", "factory_change_pct")} | {
                "off_hours_waste": bool(out["off_hours_waste"]), "new_waste": bool(out["new_waste"])}
        elif name == "asset_breakdown":
            out = an.asset_breakdown(meters, period, compare)
            brief = out[:3]
        elif name == "find_idle_load":
            out = an.idle_load(meters, ws.frames, period)
            brief = out[:3]
        elif name == "what_if_tariff":
            inp.update({k: params[k] for k in ("price_factor", "rise_kept") if k in params})
            out = an.what_if_tariff(ws.daily(compare), period, compare, rise_factor=params.get("rise_kept"),
                                    price_factor=params.get("price_factor"))
            brief = out
        elif name == "what_if_actions":
            acts = res["estimate_savings"]["actions"]
            if params.get("asset"):
                inp["asset"] = params["asset"]
                acts = [a for a in acts if a["asset"] == params["asset"]]
            out = an.what_if_actions(ws.daily(compare), period, compare, acts)
            brief = {k: out[k] for k in ("base_change_pct", "scenario_change_pct", "quarter_saving")}
        elif name == "query_validated_data":
            asset = params["asset"]
            q = ws.period_bounds(period)
            sql = f'SELECT sum("{asset}") AS kwh, sum("main") AS main_kwh FROM clean_meter WHERE ts >= ? AND ts < ?'
            row = ws.sql(sql, list(q)).iloc[0]
            out = {"meter": asset, "kwh": round(float(row["kwh"]), 1), "share_of_main": round(float(row["kwh"] / row["main_kwh"]), 4)}
            inp.update({"meter": asset, "sql": sql})
            brief = out
        elif name == "carbon_footprint":
            factor, source = ws.emission_factor()
            rows = {}
            for p in (period, compare):
                if p:
                    bounds = ws.period_bounds(p)
                    rows[p] = float(ws.sql("SELECT sum(main) AS kwh FROM clean_meter WHERE ts >= ? AND ts < ?", list(bounds)).iloc[0]["kwh"])
            out = {"kwh": round(rows[period]), "co2e_t": round(rows[period] * factor / 1000, 1), "emission_factor": factor,
                   "emission_factor_source": source}
            if compare in rows:
                out.update({"kwh_reference": round(rows[compare]), "co2e_t_reference": round(rows[compare] * factor / 1000, 1),
                            "change_pct": round(100 * (rows[period] - rows[compare]) / rows[compare], 1)})
            brief = out
        elif name == "bill_total":
            rows = {}
            for p in [x for x in (compare, period) if x]:
                d = ws.daily(compare or period)
                sel = d[an.quarter_mask(d.index, p)]
                rows[p] = {"bill": round(float(sel["cost"].sum())), "kwh": round(float(sel["kwh"].sum())),
                           "avg_price": round(float(sel["cost"].sum() / sel["kwh"].sum()), 2)}
            out = {"quarters": rows}
            brief = out
        else:  # compare_data_versions
            out = ws.changes() or {"status": "no_change"}
            brief = {k: out[k] for k in ("change", "total_before", "total_after", "cause_delta") if k in out} or out
        res[name] = out
        return {"results": res, "plan": state["plan"][1:], "loops": state["loops"] + 1,
                "steps": step(state, "analyst", name, inp, brief)}

    # ---- verify ----------------------------------------------------------------------------
    def verify(state: State) -> dict:
        res, problems = state["results"], []
        if state["loops"] >= MAX_LOOPS and state["plan"]:
            problems.append(f"plan not finished after {MAX_LOOPS} tool calls")
        b = res.get("decompose_change")
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
            if any(a[k][0] is None or a[k][0] > a[k][1] for k in keys) or not a["evidence_refs"] or a["requires_approval"] is not True:
                problems.append(f"action '{a['action']}' has an invalid range, no evidence or no approval flag")
        if res.get("compare_data_versions", {}).get("status") == "no_change":
            problems.append("no data has been loaded or corrected since the current dataset was loaded")
        if any(not math.isfinite(n) for n in _numbers({k: v for k, v in res.items() if k != "decompose_change"})):
            problems.append("a tool returned a non-finite number")
        ok = not problems
        upd: dict = {"verified": ok, "verify_note": "; ".join(problems),
                     "steps": step(state, "checker", "verify", out={"ok": ok, "problems": problems})}
        if not ok and state["replans"] < 1 and "no data has been loaded" not in upd["verify_note"]:
            # re-plan once: rebuild every derived table from the stored data, then give up honestly
            ws._derived = {}
            upd.update({"replans": state["replans"] + 1, "plan": list(PLANS[state["intent"]]), "results": {}, "loops": 0})
        return upd

    # ---- compose ---------------------------------------------------------------------------
    def compose(state: State) -> dict:
        intent, res = state["intent"], state.get("results", {})
        period, compare, params = state.get("period"), state.get("compare_to"), state.get("params", {})
        assumptions, limitations = list(state.get("assumptions", [])), list(state.get("limitations", []))
        fm = Fmt(ws.currency())
        if intent == "refuse":
            return {"answer_type": "refused", "facts": {}, "steps": step(state, "writer", "compose", out={"answer_type": "refused"}),
                    "summary": "I can't do that. Every figure here comes from the loaded data through a calculation, and every "
                               "action needs a manager's approval. Ask what the data shows instead, for example what the leak costs."}
        if intent == "out_of_scope":
            return {"answer_type": "not_in_data", "facts": {}, "steps": step(state, "writer", "compose", out={"answer_type": "not_in_data"}),
                    "summary": state.get("summary") or "The loaded data cannot answer that. It holds past meter readings, tariffs, "
                               "production and weather for the loaded quarters (no forecasts), so no figure is given."}
        if intent == "unclear":
            ask = params.get("ask") or ("Could you say what you need? For example: why the bill changed, which assets use the most "
                                        "energy, energy used when nothing is produced, what to fix first, or a tariff what-if.")
            return {"answer_type": "clarification_needed", "facts": {}, "summary": ask,
                    "steps": step(state, "writer", "compose", out={"answer_type": "clarification_needed"})}
        if not state.get("verified"):
            note = state.get("verify_note", "")
            return {"answer_type": "not_in_data", "facts": {}, "limitations": limitations + [note],
                    "summary": ("Nothing has changed since the current data was loaded, so there is no before and after to compare."
                                if "no data has been loaded" in note else "The figures could not be verified, so no number is reported. " + note),
                    "steps": step(state, "writer", "compose", out={"answer_type": "not_in_data", "reason": note})}

        facts: dict = {"period": period, "compare_to": compare}
        actions = res.get("estimate_savings", {}).get("actions", [])
        answer_type = ANSWER_TYPE[intent]
        if actions:
            factor, source = ws.emission_factor()
            assumptions += [f"Emission factor {factor} kg CO2e per kWh ({source})",
                            "Implementation costs are assumed ranges; the data has no cost table",
                            "Expected off-hours load per asset type is an engineering rule of thumb"]
            facts["actions"] = [{k: a[k] for k in ("rank", "action", "asset", "saving_kwh_per_year", "saving_money_per_year",
                                                   "saving_co2e_t_per_year", "payback_months", "implementation_cost", "ease", "rationale")}
                                for a in actions]
            facts["savings_total"] = an.savings_summary(actions)

        if intent in ("bridge", "asset_check"):
            b = res["decompose_change"]
            facts.update({"total_change_pct": b["total_change_pct"], "bill_reference": b["bill_reference"], "bill_current": b["bill_current"],
                          "uncontrollable_pct_points": b["uncontrollable_pct_points"],
                          "bridge": [{"cause": x["cause"], "pct_points": x["pct_points"], "range": x["range"]} for x in b["bridge"]]})
            waste = next(x for x in b["bridge"] if x["cause"] == "waste")
            others = sorted((x for x in b["bridge"] if x["cause"] != "waste"), key=lambda x: -abs(x["pct_points"]))
            parts = ", ".join(f"{CAUSE_LABEL[x['cause']]} {x['pct_points']:.2f}" for x in others)
            direction = "rose" if b["total_change_pct"] >= 0 else "fell"
            leaks = res.get("detect_new_waste", [])
            facts["new_waste"] = [{"asset": l["asset"], "from": l["from"], "night_kw_before": l["reference_night_kw"],
                                   "night_kw_after": round(l["reference_night_kw"] + l["excess_kw"], 1), "excess_kw": l["excess_kw"]} for l in leaks]
            lead = ""
            stated = params.get("stated_change_pct")
            if stated is not None and abs(abs(stated) - abs(b["total_change_pct"])) > 1.0:
                facts["stated_change_pct"] = stated
                lead = f"The data shows a {abs(b['total_change_pct']):.2f}% change, not {stated:g}%. "
            focus = params.get("focus_cause")
            if focus in CAUSE_LABEL:
                fx = next(x for x in b["bridge"] if x["cause"] == focus)
                share = round(100 * fx["pct_points"] / b["total_change_pct"]) if b["total_change_pct"] else 0
                facts["focus_share_pct"] = share
                lead += (f"{CAUSE_LABEL[focus].capitalize()} added {fx['pct_points']:.2f} points ({fx['range'][0]:.2f} to {fx['range'][1]:.2f}), "
                         f"about {share}% of the rise. ")
            text = (f"{lead}The bill {direction} {abs(b['total_change_pct']):.2f}% from {compare} to {period} "
                    f"({fm.money(b['bill_reference'])} to {fm.money(b['bill_current'])}). "
                    f"{b['uncontrollable_pct_points']:.2f} points are outside the factory's control: {parts}. ")
            if not b["waste_confirmed"]:
                text += (f"The remaining {waste['pct_points']:.2f} points ({waste['range'][0]:.2f} to {waste['range'][1]:.2f}) are within the "
                         f"model's error, so no new waste can be confirmed and no machine is blamed.")
            else:
                text += f"The remaining {waste['pct_points']:.2f} points ({waste['range'][0]:.2f} to {waste['range'][1]:.2f}) are new waste"
                nw = facts["new_waste"][0] if leaks else None
                text += (f": the {label(nw['asset'])} night load rose from {nw['night_kw_before']} kW to {nw['night_kw_after']} kW on "
                         f"{nw['from']} and has stayed there." if nw else ", not tied to one meter yet.")
            if intent == "asset_check":
                c = res["check_asset"]
                facts["asset_check"] = {k: c[k] for k in ("asset", "kwh", "share_of_main", "change_pct", "factory_change_pct")}
                facts["asset_share_pct"] = round(100 * c["share_of_main"], 1)
                name = label(c["asset"])
                ev = (f" {name[0].upper() + name[1:]} used {c['kwh']:,.0f} kWh ({facts['asset_share_pct']}% of the main meter), "
                      f"{'up' if (c['change_pct'] or 0) >= 0 else 'down'} {abs(c['change_pct'] or 0)}% while the whole factory changed {c['factory_change_pct']}%.")
                off = c["off_hours_waste"]
                if c["new_waste"]:
                    head = f"Partly. {name[0].upper() + name[1:]} is the one asset with new waste this quarter."
                elif off:
                    f = off[0]
                    facts["asset_off_hours"] = {"kw": f["off_hours_kw"], "excess_kw": f["excess_kw"]}
                    head = f"No. {name[0].upper() + name[1:]} did not drive the rise."
                    ev += (f" It does waste energy {_when(f)} (about {f['excess_kw'][0]}-{f['excess_kw'][1]} kW more than it needs), "
                           f"but it did the same last quarter, so that is a saving to take, not a cause of the rise.")
                else:
                    head = f"No. There is no evidence that {name} drove the rise."
                    ev += " No waste pattern was found on it."
                if params.get("asset_request") == "replace_or_upgrade":
                    answer_type = "not_in_data"
                    head = (f"The data cannot size a replacement or upgrade of {name}: it holds no equipment efficiency or price data, "
                            f"so no saving figure is given for that.")
                text = f"{head}{ev} {text}"
            elif actions:
                s = facts["savings_total"]
                text += (f" Fixing the {len(actions)} waste patterns found would save {fm.money_range(s['saving_money_per_year'])} a year; "
                         f"the first pays back in {actions[0]['payback_months'][0]}-{actions[0]['payback_months'][1]} months. Actions need a manager's approval.")
            assumptions += b["assumptions"]
            limitations += [f"Baseline fit R2 {b['baseline']['r2']} on {compare}; ranges cover fit error only, not a wrong model form"]
        elif intent == "recommend":
            lead = ""
            if params.get("asset"):
                mine = [a for a in actions if a["asset"] == params["asset"]]
                lead = (f"Yes, on {label(params['asset'])}: {mine[0]['rationale']} " if mine else
                        f"No waste pattern was found on {label(params['asset'])}. ")
            if actions:
                s = facts["savings_total"]
                lines = [f"{a['rank']}. {a['action']}: {fm.money_range(a['saving_money_per_year'])} a year, payback "
                         f"{a['payback_months'][0]}-{a['payback_months'][1]} months ({a['ease']})" for a in actions[:3]]
                text = (f"{lead}{len(actions)} actions would save {fm.money_range(s['saving_money_per_year'])} a year "
                        f"({fm.num(s['saving_kwh_per_year'][0] / 1000)}-{fm.num(s['saving_kwh_per_year'][1] / 1000)} MWh, "
                        f"{s['saving_co2e_t_per_year'][0]}-{s['saving_co2e_t_per_year'][1]} t CO2e) for {fm.money_range(s['implementation_cost'])} of work. "
                        f"Ranked by payback: " + "; ".join(lines) + ". Each is a suggestion for a manager to approve.")
                limitations.append("Payback depends on the assumed implementation costs")
            else:
                text = (f"{lead}No controllable waste pattern was found in {period}, so no action is recommended. Tariff, volume, mix and "
                        f"weather are not things the factory can fix this way.")
        elif intent == "top_users":
            rows = res["asset_breakdown"]
            facts["assets"] = rows
            top = [r for r in rows if r["asset"] != "unmetered"][:3]
            facts["shares_pct"] = [round(100 * r["share_of_main"], 1) for r in rows]
            text = (f"In {period} the biggest users were " + ", ".join(f"{r['asset']} ({100 * r['share_of_main']:.1f}% of the main meter)" for r in top) + ". ")
            unmetered = next(r for r in rows if r["asset"] == "unmetered")
            text += f"{100 * unmetered['share_of_main']:.1f}% of the main meter has no sub-meter; it is reported as unmetered, not as waste."
        elif intent == "idle":
            rows = res["find_idle_load"]
            if params.get("asset"):
                rows = [r for r in rows if r["asset"] == params["asset"]] or rows
            facts["idle"] = rows
            total = round(sum(r["idle_kwh"] for r in rows), 1)
            facts["idle_total_kwh"] = total
            facts["idle_share_pct"] = [round(100 * r["idle_share_of_asset"]) for r in rows]
            days = rows[0]["non_working_days"] if rows else 0
            text = (f"When nothing is produced in {period} (nights, and {days} non-working days), the sub-meters used {total:,.0f} kWh. Largest: "
                    + ", ".join(f"{r['asset']} {r['idle_kwh']:,.0f} kWh ({round(100 * r['idle_share_of_asset'])}% of its energy, busiest at "
                                f"{r['busiest_idle_hour']:02d}:00)" for r in rows[:3]) + ".")
            off = [f for f in res.get("find_off_hours_waste", []) if not params.get("asset") or f["asset"] == params["asset"]]
            if off:
                facts["off_hours"] = [{"asset": f["asset"], "excess_kw": f["excess_kw"]} for f in off]
                text += " More than these assets need: " + "; ".join(f"{f['asset']} {_when(f)} (about {f['excess_kw'][0]}-{f['excess_kw'][1]} kW too much)" for f in off) + "."
            limitations.append("Idle means outside the 06:00-22:00 shift on working days, and all of a non-working day")
        elif intent == "what_if_tariff":
            w = res["what_if_tariff"]
            facts.update(w)
            if params.get("price_factor") is not None:
                pf = params["price_factor"]
                facts["price_change_pct"] = round(100 * abs(pf - 1), 1)
                scen = f"with every current price {facts['price_change_pct']:g}% {'lower' if pf < 1 else 'higher'}"
            else:
                facts["price_rise_kept_pct"] = round(100 * params["rise_kept"], 1)
                scen = "with no tariff change" if params["rise_kept"] == 0 else f"with {facts['price_rise_kept_pct']:g}% of the price rise"
            text = (f"{scen.capitalize()}, the {period} bill would be {fm.money(w['bill_scenario'])} instead of {fm.money(w['bill_current'])}, "
                    f"a change of {w['scenario_change_pct']:.2f}% on {compare} instead of {w['base_change_pct']:.2f}%.")
            assumptions.append("Energy use is unchanged in the scenario; only the price differs")
        elif intent == "what_if_actions":
            w = res["what_if_actions"]
            facts.update(w)
            if params.get("asset"):
                facts["actions"] = [a for a in facts.get("actions", []) if a["action"] in w["modelled_actions"]]
                facts["savings_total"] = an.savings_summary([a for a in actions if a["action"] in w["modelled_actions"]])
            what = (f"the {label(params['asset'])} fix ({w['modelled_actions'][0].lower()})" if params.get("asset") and w["modelled_actions"]
                    else f"the {len(w['modelled_actions'])} recommended actions")
            text = (f"Had {what} been in place for all of {period}, the bill would have been about "
                    f"{fm.money_range(w['quarter_saving'])} lower, a change of {w['scenario_change_pct'][0]:.2f}% to {w['scenario_change_pct'][1]:.2f}% "
                    f"on {compare} instead of {w['base_change_pct']:.2f}%." if w["modelled_actions"] else
                    f"No waste was found in {period}, so there is no action to model.")
        elif intent == "meter_query":
            m = res["query_validated_data"]
            facts.update(m)
            facts["share_pct"] = round(100 * m["share_of_main"], 1)
            text = f"The {m['meter']} meter used {m['kwh']:,.0f} kWh in {period}, {facts['share_pct']}% of the main meter."
        elif intent == "footprint":
            f = res["carbon_footprint"]
            facts.update(f)
            text = (f"In {period} the factory used {f['kwh']:,} kWh, about {f['co2e_t']:,} t CO2e at {f['emission_factor']} kg CO2e per kWh "
                    f"({f['emission_factor_source']}).")
            if "co2e_t_reference" in f:
                text += f" That is {abs(f['change_pct'])}% {'more' if f['change_pct'] >= 0 else 'less'} than {compare} ({f['co2e_t_reference']:,} t)."
            assumptions.append(f"Emission factor {f['emission_factor']} kg CO2e per kWh ({f['emission_factor_source']})")
        elif intent == "bill_total":
            rows = res["bill_total"]["quarters"]
            facts["quarters"] = rows
            text = " ".join(f"The {p} bill was {fm.money(r['bill'])} for {r['kwh']:,} kWh (average {r['avg_price']} per kWh)." for p, r in rows.items())
        else:  # changes
            c = res["compare_data_versions"]
            facts.update({k: c[k] for k in ("total_before", "total_after", "cause_delta", "asset_kwh_change_pct", "savings_before",
                                             "savings_after", "bill_before", "bill_after")})
            moved = [f"{CAUSE_LABEL[k]} {v:+.2f} points" for k, v in c["cause_delta"].items() if abs(v) >= 0.01]
            text = (f"After the latest data ({c['change']}), the bill change for {period} moved from {c['total_before']:.2f}% to "
                    f"{c['total_after']:.2f}%. " + (f"What moved: {', '.join(moved)}; the other causes are unchanged. " if moved else
                                                    "No cause moved by 0.01 points or more. "))
            if c["asset_kwh_change_pct"]:
                text += "Metered energy changed for " + ", ".join(f"{a} ({v:+.1f}%)" for a, v in c["asset_kwh_change_pct"].items()) + ". "
            text += f"Savings found are now {fm.money_range(c['savings_after'])} a year (were {fm.money_range(c['savings_before'])})."
        facts["display"] = fm.shown

        upd: dict = {"answer_type": answer_type, "facts": facts, "summary": text, "assumptions": assumptions, "limitations": limitations}
        used, info = "template", {}
        if intent in ("bridge", "asset_check", "recommend", "what_if_actions", "changes") and answer_type != "not_in_data" \
                and llm.provider() != "mock":
            r = llm.reword(text, facts)
            info = {"model": r["model"], "ms": r["ms"], "tokens": r["tokens"]}
            upd["llm_calls"], upd["tokens"] = state.get("llm_calls", 0) + 1, state.get("tokens", 0) + r["tokens"]
            if r["error"]:
                upd["llm_errors"] = state.get("llm_errors", []) + [f"brief: {r['error']}"]
            if r["text"]:
                upd["summary"], used = r["text"], "llm"
                upd["facts"] = {**facts, "_template": text}
        upd["steps"] = step(state, "writer", "write_finance_brief", {"writer": used, **info}, {"answer_type": answer_type})
        return upd

    # ---- number check ----------------------------------------------------------------------
    def number_check(state: State) -> dict:
        if state.get("answer_type") in ("clarification_needed", "refused"):  # fixed wording, no data figures
            return {"steps": step(state, "checker", "number_check", out={"skipped": "no data figures in this answer",
                                                                           "all_numbers_match_tools": True})}
        facts = state.get("facts", {})
        clean_facts = {k: v for k, v in facts.items() if k != "_template"}
        bad = unmatched_numbers(state["summary"], clean_facts)
        upd: dict = {}
        if bad and "_template" in facts:  # the LLM wording carried a number no tool produced: use the template
            upd["summary"] = facts["_template"]
            upd["limitations"] = state.get("limitations", []) + ["LLM wording was rejected by the number check; template wording used"]
            bad_after = unmatched_numbers(upd["summary"], clean_facts)
        else:
            bad_after = bad
        if bad_after:
            upd.update({"answer_type": "not_in_data", "summary": "The answer was withheld because a figure in it could not be traced to a tool result."})
        upd["steps"] = step(state, "checker", "number_check", out={"unmatched_first_pass": bad, "unmatched_final": bad_after,
                                                                   "all_numbers_match_tools": not bad_after})
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
