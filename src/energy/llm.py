"""The only place an LLM is called. It does two jobs: read a question into a structured request
(intent and parameters), and reword a brief written from tool results. It never calculates.

LLM_PROVIDER = mock | bedrock | anthropic   (MOCK_LLM=true forces mock)
  bedrock    Claude on Amazon Bedrock through an AWS profile (AWS_PROFILE, AWS_REGION)
  anthropic  Claude API (ANTHROPIC_API_KEY)
LLM_MODEL    model id; for Bedrock the default is Claude Haiku 4.5.
Any failure falls back to the deterministic path, and the error is reported in the trace.
"""
from __future__ import annotations

import json
import os
import time

INTENTS = ["bridge", "asset_check", "top_users", "idle", "recommend", "what_if_tariff", "what_if_actions",
           "meter_query", "footprint", "changes", "bill_total", "out_of_scope", "refuse", "unclear"]
DEFAULT_MODEL = {"bedrock": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic": "claude-haiku-4-5"}
_client = None
_health: dict = {}


def provider() -> str:
    if os.getenv("MOCK_LLM", "").lower() in ("1", "true", "yes"):
        return "mock"
    return os.getenv("LLM_PROVIDER", "mock").lower()


def model() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_MODEL.get(provider(), "")


def _get_client():
    global _client
    if _client is None:
        import anthropic

        if provider() == "bedrock":
            _client = anthropic.AnthropicBedrock(aws_profile=os.getenv("AWS_PROFILE") or None,
                                                 aws_region=os.getenv("AWS_REGION", "us-east-1"), max_retries=1, timeout=30)
        else:
            _client = anthropic.Anthropic(max_retries=1, timeout=30)
    return _client


def call(system: str, user: str, max_tokens: int, schema: dict | None = None) -> dict:
    """One model call. Returns {text, tokens, ms, error}; text is None when the model is
    unavailable, refuses, or the provider is mock."""
    out = {"text": None, "tokens": 0, "ms": 0, "error": None, "model": model()}
    if provider() not in ("bedrock", "anthropic"):
        return out
    t0 = time.perf_counter()
    try:
        kwargs = {"output_config": {"format": {"type": "json_schema", "schema": schema}}} if schema else {}
        resp = _get_client().messages.create(model=model(), max_tokens=max_tokens, system=system,
                                             messages=[{"role": "user", "content": user}], **kwargs)
        out["tokens"] = resp.usage.input_tokens + resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            out["error"] = "model declined"
        elif resp.stop_reason == "max_tokens":
            out["error"] = "output cut off at max_tokens"
        else:
            out["text"] = "".join(b.text for b in resp.content if b.type == "text").strip() or None
    except Exception as e:  # no credentials, network, throttling: the caller uses the deterministic path
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    out["ms"] = round(1000 * (time.perf_counter() - t0))
    return out


def status(ping: bool = True) -> dict:
    """Provider status. With ping, one tiny call (cached for five minutes) proves the model answers."""
    p = provider()
    if p == "mock":
        return {"provider": "mock", "status": "ok", "detail": "template briefs, keyword routing"}
    if p not in ("bedrock", "anthropic"):
        return {"provider": p, "status": "not implemented; using mock"}
    info = {"provider": p, "model": model()}
    if p == "bedrock":
        info.update({"aws_profile": os.getenv("AWS_PROFILE", "default"), "aws_region": os.getenv("AWS_REGION", "us-east-1")})
    if not ping:
        return {**info, "status": "configured"}
    if time.time() - _health.get("at", 0) > 300:
        r = call("Reply with the single word: ok", "ping", 10)
        _health.update({"at": time.time(), "ok": r["text"] is not None, "ms": r["ms"], "error": r["error"]})
    return {**info, "status": "ok" if _health["ok"] else "unavailable; using templates",
            "latency_ms": _health["ms"], **({"error": _health["error"]} if _health["error"] else {})}


ROUTE_SYSTEM = """You read one question asked of a factory energy-cost analysis tool and return a structured request.
The tool has two quarters of 15-minute meter data (main meter and sub-meters), tariffs, production and weather.
It can explain why the bill changed, find waste and recommend savings. It cannot forecast or invent figures.

intent, choose one:
- bridge: why the bill or cost changed between quarters, what caused the increase, how much one cause (tariff, weather, volume, mix) contributed, or a comparison of two quarters.
- asset_check: whether a named asset caused the bill rise, or the saving from replacing or upgrading a named asset (never refuse these: the tool explains what the data can and cannot show).
- top_users: which assets use the most energy, or a breakdown of energy by asset.
- idle: energy used when nothing is produced, including questions limited to nights, weekends, Sundays, holidays or idle hours, for all assets or one asset.
- recommend: what to do first, savings opportunities, whether there is a leak or waste on an asset, ranking actions by payback or ease.
- what_if_tariff: a hypothetical tariff or price (lower, higher, unchanged, a different rise).
- what_if_actions: what the bill would be if the recommended actions, or one named fix such as the leak repair, were done.
- meter_query: total energy of one named meter in a quarter, with no time-of-day or day-type limit.
- footprint: CO2 or carbon emissions.
- bill_total: the size of the bill or energy cost in a quarter.
- changes: what changed after new or corrected data was loaded (the late data drop).
- out_of_scope: forecasts or future prices, other sites, hypothetical changes in production volume, weather or mix, or anything else not answerable from the data.
- refuse: only when it explicitly asks to invent, guess or make up a number, skip checks, or approve an action.
- unclear: only when no intent above fits at all. When in doubt, pick the closest intent: any question about why or how
  a cause (tariff, weather, volume, mix, waste) affected the bill is bridge with focus_cause.

Fields:
- period, compare_to: quarters as YYYY-Qn when the question names them, else null.
- asset: the meter the question is about, exactly as listed in METERS, else null. Map plain words (lights -> lighting, air compressor -> compressors, AC or cooling -> hvac, line 2 -> line_2).
- asset_request: "replace_or_upgrade" if it asks about replacing or upgrading the asset, "blame" if it asks whether the asset caused the rise or wastes energy, else null.
- price_factor: for what_if_tariff, the multiplier on current prices (10% lower -> 0.9, 5% higher -> 1.05), else null.
- rise_kept: for what_if_tariff about the price rise itself, the share of the rise kept (no rise -> 0, half the rise -> 0.5), else null.
- focus_cause: for bridge, the one cause the question asks about (tariff, volume, product_mix, weather, waste), else null.
- stated_change_pct: only when the question states how much the bill changed (for example "up 15%" -> 15), else null.
- clarification: only when intent is unclear, one short question to ask back, else null."""


def route(question: str, meters: list[str], quarters: list[str]) -> dict:
    nullable = lambda t: {"anyOf": [t, {"type": "null"}]}
    schema = {"type": "object", "additionalProperties": False,
              "required": ["intent", "period", "compare_to", "asset", "asset_request", "price_factor", "rise_kept",
                           "focus_cause", "stated_change_pct", "clarification"],
              "properties": {"intent": {"type": "string", "enum": INTENTS},
                             "period": nullable({"type": "string"}), "compare_to": nullable({"type": "string"}),
                             "asset": nullable({"type": "string", "enum": meters}) if meters else {"type": "null"},
                             "asset_request": nullable({"type": "string", "enum": ["replace_or_upgrade", "blame"]}),
                             "price_factor": nullable({"type": "number"}), "rise_kept": nullable({"type": "number"}),
                             "focus_cause": nullable({"type": "string", "enum": ["tariff", "volume", "product_mix", "weather", "waste"]}),
                             "stated_change_pct": nullable({"type": "number"}), "clarification": nullable({"type": "string"})}}
    user = f"METERS: {', '.join(meters)}\nQUARTERS IN DATA: {', '.join(quarters)}\nQUESTION: {question}"
    r = call(ROUTE_SYSTEM, user, 400, schema)
    try:
        r["request"] = json.loads(r["text"]) if r["text"] else None
    except ValueError:
        r["request"], r["error"] = None, "route output was not valid JSON"
    return r


BRIEF_SYSTEM = """You write a short brief for a factory finance manager. Rewrite the DRAFT in plain, confident language,
at most 110 words, no headings or bullet points. Lead with the answer.
Use only numbers that appear in FACTS, written exactly as they appear there (same rounding), and keep every range.
Do not add causes, advice or numbers that are not in FACTS. Do not blame a machine for anything FACTS attributes to
tariff, volume, product mix or weather. Keep the DRAFT's first sentence's message: if it answers yes or no, or corrects a
figure stated in the question, start with that. Write money exactly as the DRAFT does (for example "₹6.9-8.3 lakh").
Say that actions need a manager's approval when the DRAFT does."""


def reword(template: str, facts: dict) -> dict:
    return call(BRIEF_SYSTEM, f"FACTS:\n{json.dumps(facts, default=str)}\n\nDRAFT:\n{template}", 600)
