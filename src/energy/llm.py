"""The only place an LLM is called. It does two jobs: label a request the keyword router
could not place, and reword a brief. It never calculates.

LLM_PROVIDER = mock (default) | anthropic. MOCK_LLM=true forces mock. Any failure falls
back to the deterministic path, so the API works with no key and no network.
"""
from __future__ import annotations

import json
import os

LABELS = ["bridge", "top_users", "idle", "recommend", "what_if", "meter_query", "out_of_scope", "refuse", "unclear"]


def provider() -> str:
    if os.getenv("MOCK_LLM", "").lower() in ("1", "true", "yes"):
        return "mock"
    return os.getenv("LLM_PROVIDER", "mock").lower()


def model() -> str:
    return os.getenv("LLM_MODEL", "claude-opus-5-5")


def status() -> dict:
    p = provider()
    if p == "mock":
        return {"provider": "mock", "status": "ok", "detail": "template briefs, keyword routing"}
    if p == "anthropic":
        has = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
        return {"provider": p, "model": model(), "status": "configured" if has else "no credentials in environment"}
    return {"provider": p, "status": "not implemented; falling back to mock"}


def _call(system: str, user: str, max_tokens: int) -> tuple[str | None, int]:
    """Returns (text, tokens). text is None when the model is unavailable or declines."""
    if provider() != "anthropic":
        return None, 0
    try:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.beta.messages.create(
            model=model(), max_tokens=max_tokens, system=system,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=[{"role": "user", "content": user}],
        )
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            return None, tokens
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return (text or None), tokens
    except Exception:  # no key, network down, rate limit: the caller uses the template
        return None, 0


def classify(question: str) -> tuple[str | None, int]:
    system = ("You label a question asked of a factory energy-cost analysis tool. Reply with exactly one label "
              f"from this list and nothing else: {', '.join(LABELS)}.\n"
              "bridge: why the bill changed. top_users: which assets use most energy. idle: energy used with no "
              "production. recommend: what to fix, savings, payback. what_if: a hypothetical tariff. meter_query: "
              "energy of one named meter. out_of_scope: forecasts or anything not answerable from two quarters of "
              "meter, tariff, production and weather data. refuse: asks to invent numbers or skip checks. "
              "unclear: too vague to act on.")
    text, tokens = _call(system, question, 200)
    label = (text or "").strip().lower()
    return (label if label in LABELS else None), tokens


def reword(template: str, facts: dict) -> tuple[str | None, int]:
    system = ("You write a short brief for a factory finance manager. Rewrite the DRAFT in plain language, at most "
              "120 words, no headings or bullet points. Use only numbers that appear in FACTS, written exactly as "
              "they appear there, and keep every range. Do not add causes, advice or numbers that are not in FACTS. "
              "Do not blame a machine for anything FACTS attributes to tariff, volume, product mix or weather.")
    return _call(system, f"FACTS:\n{json.dumps(facts, default=str)}\n\nDRAFT:\n{template}", 2000)
