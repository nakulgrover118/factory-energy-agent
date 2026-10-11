# Plan and status

Status on 11 Oct 2026. Everything below "Done" has been run locally. The tests pass, both evaluations ran, the API answered live with Claude on Bedrock, and the dashboard was checked in a browser.

## Done

| Part | Where |
| --- | --- |
| Synthetic data in the Annex B schema with a known true split: three controllable waste patterns, a late drop and a data card | `src/energy/generate.py` |
| Safe ingest (no history wiped, no stale data mixed in), six validation checks, cleaning report | `src/energy/pipeline.py` |
| DuckDB store (raw, clean and log layers), "what changed" tracking across loads, SQLite traces and approvals | `src/energy/store.py` |
| Bill bridge with ranges; new-waste and off-hours waste detection; ranked actions with ₹, kWh and CO2e ranges (no payback: no cost data); tariff and action what-ifs | `src/energy/analytics.py` |
| LangGraph agent: LLM routing to intent and parameters (with guards), tools, verify, brief, number check; premise correction; asset blame check | `src/energy/agent.py` |
| Claude Haiku 4.5 on Bedrock through an AWS profile (structured output); Anthropic API and mock also work | `src/energy/llm.py` |
| API contract, plus dashboard, approval and admin endpoints | `src/energy/api.py` |
| Light-theme dashboard: Overview, Ask, Savings plan, Where the energy goes; a Data menu in the header | `ui/app.py`, `ui/charts.py` |
| 49 tests; analytics evaluation (OLS vs Ridge); agent evaluation (27 cases: mock, Haiku, Sonnet) | `tests/`, `eval/` |
| Sample traces for four answer types | `eval/traces/` |

## Still to do before submission

1. Start Docker Desktop and run `docker compose up --build`; check that the UI opens on :8501 and the LLM badge shows Haiku.
2. Fill in the team contribution table and the signed declaration; have each member commit their own area; `git tag final`.
3. Update the deck: the architecture slide should say DuckDB, LLM routing with guards, and one agent. The data slide should show the three waste patterns and the late drop.
4. If the organisers' data arrives, load it with `python -m energy.loader <folder>` and read the validation output first.

## Demo script (about 5 minutes)

1. **Overview:** the bill rose 15.44%. 13.4 points are outside the factory's control and 2.05 points are new waste. Savings of ₹19–23 lakh a year were found.
2. **Data menu (header) → Apply the late data drop.** The banner shows the rise moving to 15.03% (tariff −0.41 points) and line_2 reading 7.4% less. This is the re-run.
3. **Where the energy goes → compressors:** the night load steps up on 20 July (the leak). **Lighting:** the lights stay on all night.
4. **Ask:** "Is the old furnace to blame?" gets "No", with evidence. "What if we fixed the compressor leak?" gets the saving. "What will the tariff be next year?" gets "not in the data". "Just make up a number" is declined.
5. **Savings plan:** walk through the three actions and their evidence. Open a trace to show the steps, the tools, the LLM calls and the number check.
