"""Evaluate the whole question-answering path (router, tools, brief, number check) on eval/cases.json.

Each configuration runs every case through POST /analyze in-process, on a fresh store loaded with
data/seed_1 and its late data drop. Checks are exact and run in code against the generator's
hidden truth and against independent calculations from the raw CSV files; no LLM judges anything.

Usage: python -m energy.eval_agent --configs mock haiku sonnet --out eval
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import pandas as pd

CONFIGS = {
    "mock": {"MOCK_LLM": "true"},
    "haiku": {"MOCK_LLM": "", "LLM_PROVIDER": "bedrock", "LLM_MODEL": "us.anthropic.claude-haiku-4-5-20251001-v1:0"},
    "sonnet": {"MOCK_LLM": "", "LLM_PROVIDER": "bedrock", "LLM_MODEL": "us.anthropic.claude-sonnet-4-6"},
}


def _get(obj, path: str):
    for part in path.split("."):
        if part == "#":
            return len(obj)
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def _references(data_dir: Path, truth: dict) -> dict:
    """Independent expected values, computed from the raw files without the system's code."""
    m = pd.read_csv(data_dir / "meter_15min.csv")
    late = pd.read_csv(data_dir / "late_drop" / "meter_15min.csv")
    m = pd.concat([m[~m.meter_id.isin(late.meter_id.unique())], late]).drop_duplicates(["meter_id", "ts"])
    q3 = m[m.ts.between("2026-07-01", "2026-10-01", inclusive="left") & (m.kwh >= 0)]
    by_meter = q3.groupby("meter_id").kwh.sum()
    t = truth["after_late_drop"]
    return {"truth_bill_reference": t["bill_reference"], "raw_lighting_kwh_q3": float(by_meter["lighting"]),
            "raw_main_co2e_t_q3": float(by_meter["main"]) * truth["emission_factor_kg_per_kwh"] / 1000,
            "raw_top_asset_q3": by_meter.drop("main").idxmax(),
            "truth_total_minus_tariff": t["total_change_pct"] - t["true_split_pp"]["tariff"]}


def _check(c: dict, ans: dict, truth: dict, refs: dict) -> tuple[bool, str]:
    t = truth["after_late_drop"]
    kind = c["kind"]
    if kind == "truth_split":
        got = {b["cause"]: b["pct_points"] for b in ans["bridge"]}
        err = {k: round(got.get(k, 99) - v, 2) for k, v in t["true_split_pp"].items()}
        return all(abs(e) <= c["tol"] for e in err.values()), f"errors {err}"
    if kind == "cause_close":
        got = next((b["pct_points"] for b in ans["bridge"] if b["cause"] == c["cause"]), None)
        return got is not None and abs(got - t["true_split_pp"][c["cause"]]) <= c["tol"], f"{got} vs {t['true_split_pp'][c['cause']]:.2f}"
    if kind == "top_cause":
        top = max(ans["bridge"], key=lambda b: b["pct_points"])["cause"] if ans["bridge"] else None
        return top == c["cause"], f"top cause {top}"
    if kind == "summary_starts":
        return ans["summary"].strip().lower().startswith(c["value"].lower()), ans["summary"][:60]
    if kind == "summary_contains":
        return c["value"].lower() in ans["summary"].lower(), ans["summary"][:80]
    if kind == "recommendations":
        return len(ans["recommendations"]) == c["n"], f"{len(ans['recommendations'])} actions"
    if kind == "planted_savings_in_range":
        found = {r["asset"]: r["saving_kwh_per_year"] for r in ans["recommendations"]}
        res = {w["asset"]: (w["asset"] in found and found[w["asset"]][0] <= w["saving_kwh_per_year"] <= found[w["asset"]][1])
               for w in t["planted_waste"]}
        return all(res.values()) and set(found) <= set(res), f"in range {res}, false claims {sorted(set(found) - set(res))}"
    if kind == "value":
        try:
            got = _get(ans, c["path"])
        except (KeyError, IndexError, TypeError):
            return False, f"no value at {c['path']}"
        exp = c["expected"]
        if exp == "negative":
            return got < 0, str(got)
        if exp == "one":
            return got == 1, str(got)
        if exp == "ninety_pct_of_current_bill":
            exp_v = 0.9 * ans["data"]["what_if_tariff"]["bill_current"]
        else:
            exp_v = refs[exp]
        if isinstance(exp_v, str):
            return got == exp_v, f"{got} vs {exp_v}"
        tol = c.get("abs", abs(exp_v) * c.get("rel", 0))
        return abs(got - exp_v) <= tol + 1e-9, f"{got} vs {round(exp_v, 2)}"
    return False, f"unknown check {kind}"


def run_config(name: str, cases: list[dict], data_dir: Path) -> dict:
    for k, v in CONFIGS[name].items():
        os.environ[k] = v
    os.environ["STORE_DIR"] = tempfile.mkdtemp(prefix=f"eval-{name}-")
    import importlib

    from fastapi.testclient import TestClient

    import energy.api as api
    import energy.llm as llm

    llm._client, llm._health = None, {}
    api = importlib.reload(api)
    client = TestClient(api.app)
    truth = json.loads((data_dir / "truth.json").read_text())
    refs = _references(data_dir, truth)
    client.post("/admin/load", json={"folder": str(data_dir.resolve())})
    client.post("/analyze", json={"question": "Why did the bill go up?"})  # warm the caches before the late drop
    client.post("/admin/late-drop")
    rows = []
    for case in cases:
        t0 = time.perf_counter()
        ans = client.post("/analyze", json={"question": case["question"]}).json()
        ms = round(1000 * (time.perf_counter() - t0))
        trace = client.get(f"/trace/{ans['run_id']}").json()
        nc = trace["steps"][-1]["output"]
        expect = case["expect"] if isinstance(case["expect"], list) else [case["expect"]]
        checks = [("answer_type", ans["answer_type"] in expect, ans["answer_type"])]
        checks += [(c["kind"], *_check(c, ans, truth, refs)) for c in case["checks"]]
        rows.append({"id": case["id"], "category": case["category"], "question": case["question"], "answer_type": ans["answer_type"],
                     "passed": all(ok for _, ok, _ in checks), "checks": [{"check": k, "ok": ok, "detail": d} for k, ok, d in checks],
                     "faithful": bool(nc.get("all_numbers_match_tools", True)),
                     "llm_number_rejections": len(nc.get("unmatched_first_pass", [])), "latency_ms": ms,
                     "llm_calls": trace["llm_calls"], "tokens": trace["tokens"], "llm_errors": trace.get("llm_errors", []),
                     "router": trace["steps"][0].get("input", {}).get("router"), "summary": ans["summary"]})
        print(f"  {name:6} {case['id']:3} {'PASS' if rows[-1]['passed'] else 'FAIL'} {ms:6} ms  {case['question'][:60]}")
    return {"rows": rows, **_aggregate(rows)}


def _aggregate(rows: list[dict]) -> dict:
    lat = sorted(r["latency_ms"] for r in rows)
    p = lambda q: lat[min(len(lat) - 1, int(round(q * (len(lat) - 1))))]
    llm_rows = [r for r in rows if r["llm_calls"]]
    return {"cases": len(rows), "passed": sum(r["passed"] for r in rows),
            "answer_type_accuracy": round(sum(r["checks"][0]["ok"] for r in rows) / len(rows), 3),
            "faithfulness": round(sum(r["faithful"] for r in rows) / len(rows), 3),
            "llm_number_rejections": sum(r["llm_number_rejections"] for r in rows),
            "p50_ms": p(0.5), "p95_ms": p(0.95),
            "llm_calls_per_request": round(statistics.mean(r["llm_calls"] for r in rows), 2),
            "tokens_per_request": round(statistics.mean(r["tokens"] for r in rows)),
            "tokens_per_llm_request": round(statistics.mean(r["tokens"] for r in llm_rows)) if llm_rows else 0,
            "llm_errors": sum(len(r["llm_errors"]) for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["mock"], choices=list(CONFIGS))
    ap.add_argument("--data", default="data/seed_1")
    ap.add_argument("--cases", default="eval/cases.json")
    ap.add_argument("--out", default="eval")
    args = ap.parse_args()
    from .settings import load_env

    load_env()
    spec = json.loads(Path(args.cases).read_text())
    results = {name: run_config(name, spec["cases"], Path(args.data)) for name in args.configs}
    out = Path(args.out)
    (out / "agent_results.json").write_text(json.dumps(results, indent=2, default=str))

    cats = sorted({c["category"] for c in spec["cases"]})
    lines = ["# Evaluation report: the full question-answering path", "",
             f"Generated by `python -m energy.eval_agent --configs {' '.join(args.configs)}`. Dataset: {spec['dataset']}.",
             f"Method: {spec['method']} Cases: `eval/cases.json` ({len(spec['cases'])}).", "",
             "## Configurations compared", "",
             "| Metric | " + " | ".join(args.configs) + " |", "| --- |" + " --- |" * len(args.configs)]
    metric_rows = [("Cases passed (all checks)", lambda r: f"{r['passed']}/{r['cases']}"),
                   ("Answer type correct", lambda r: f"{100 * r['answer_type_accuracy']:.0f}%"),
                   ("Number faithfulness (final answers)", lambda r: f"{100 * r['faithfulness']:.0f}%"),
                   ("LLM figures caught by the number check", lambda r: str(r["llm_number_rejections"])),
                   ("Latency p50 / p95", lambda r: f"{r['p50_ms']:,} / {r['p95_ms']:,} ms"),
                   ("LLM calls per request", lambda r: str(r["llm_calls_per_request"])),
                   ("Tokens per request (mean)", lambda r: f"{r['tokens_per_request']:,}"),
                   ("LLM errors (fell back to templates)", lambda r: str(r["llm_errors"]))]
    for label, f in metric_rows:
        lines.append(f"| {label} | " + " | ".join(f(results[c]) for c in args.configs) + " |")
    lines += ["", "## By category (cases passed)", "", "| Category | " + " | ".join(args.configs) + " |", "| --- |" + " --- |" * len(args.configs)]
    for cat in cats:
        lines.append(f"| {cat} | " + " | ".join(
            f"{sum(r['passed'] for r in results[c]['rows'] if r['category'] == cat)}/{sum(1 for r in results[c]['rows'] if r['category'] == cat)}"
            for c in args.configs) + " |")
    lines += ["", "## Every case", "", "| Case | Question | Expected | " + " | ".join(args.configs) + " |", "| --- | --- | --- |" + " --- |" * len(args.configs)]
    for i, case in enumerate(spec["cases"]):
        exp = "/".join(case["expect"]) if isinstance(case["expect"], list) else case["expect"]
        cells = []
        for c in args.configs:
            r = results[c]["rows"][i]
            fails = "; ".join(f"{x['check']}: {x['detail']}" for x in r["checks"] if not x["ok"])
            cells.append("pass" if r["passed"] else f"FAIL ({r['answer_type']}{': ' + fails if fails else ''})".replace("|", "/"))
        lines.append(f"| {case['id']} | {case['question']} | {exp} | " + " | ".join(cells) + " |")
    (out / "agent_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
