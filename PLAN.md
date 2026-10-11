# Implementation plan: Factory Energy Cost Decomposition and Savings Agent

Status on 11 Oct 2026: data, analytics, the FastAPI contract and the LangGraph agent are built and tested (32 tests). The Streamlit UI is built and checked against the live API. The Dockerfile and docker-compose.yml are written and the compose file validates, but no image has been built because the Docker daemon was not running. README and the LLM-path evaluation are not built yet. The Anthropic provider is written but has never been called with a real key.

## 1. What exists now

| Part | File | State |
| --- | --- | --- |
| Synthetic data generator with known true split | `src/energy/generate.py` | Done |
| Validation (six checks), cleaning, ingest merge | `src/energy/pipeline.py` | Done |
| Pricing, baseline, bill bridge, ranges, asset breakdown, idle load, leak detection, savings, what-if | `src/energy/analytics.py` | Done |
| Evaluation against truth, two configurations | `src/energy/evaluate.py`, `eval/report.md` | Done |
| Workspace, CSV store, SQLite trace | `src/energy/store.py` | Done |
| LangGraph agent: route, tool loop (cap 4), verify, compose, number check | `src/energy/agent.py` | Done |
| LLM layer: mock by default, Anthropic optional | `src/energy/llm.py` | Done, untested against the real API |
| API: `/analyze`, `/ingest`, `/health`, `/trace/{run_id}`, `/validation` | `src/energy/api.py` | Done |
| Loader CLI | `src/energy/loader.py` | Done |
| Streamlit UI (calls the API only) | `ui/app.py` | Done |
| Docker: one image, `api` and `ui` services, demo data on first start | `Dockerfile`, `docker-compose.yml`, `src/energy/bootstrap.py` | Written, not built or run |
| Tests (32) | `tests/test_core.py`, `tests/test_api.py` | Passing |
| Datasets | `data/seed_1..3`, `data/control`, `data/stress` | Generated |

Run it:

```
.venv\Scripts\python -m pip install -r requirements.txt
set PYTHONPATH=src
.venv\Scripts\python -m energy.generate --out data --seeds 1 2 3 --control --stress
.venv\Scripts\python -m energy.evaluate --data data --out eval
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m energy.loader data\seed_1
.venv\Scripts\python -m uvicorn energy.api:app --app-dir src --port 8000
```

## 2. What was checked

| Check | Result |
| --- | --- |
| Python | 3.11.8 available (`py -3.11`); a project `.venv` holds the packages |
| Docker, Git | Installed (Docker 29.8, Git 2.50); no repository initialised yet |
| Packages for the API and agent | Installed in `.venv` and pinned in `requirements.txt` |
| LLM key | None found; `MOCK_LLM` mode is needed until one is supplied |
| HCLTech data pack | Not available; all results are on our own generated data |

## 3. Changes proposed to the design

| Design doc says | Change | Why |
| --- | --- | --- |
| Baseline regresses daily kWh | Regress daily **cost at reference prices** | Peak and off-peak weighting is handled exactly; no separate price weight is needed |
| Waste = actual minus expected energy | Waste = bill change minus tariff and the three driver effects | Reconciles by construction for any baseline, including Ridge |
| Ridge as fallback | Keep OLS as default; Ridge only if coefficients turn negative | Ridge was worse on every dataset (mean error 0.15 vs 0.09 points) |
| Ranges from bootstrap | Keep, but state they cover fit error only | On the stress set the truth fell outside 2 of 5 ranges |
| Tariff schema as documented | Accept optional `start_hour`, `end_hour`; assume peak 06-22 when absent | The documented schema does not say which hours are peak |
| Calendar effect unspecified | Count working-day changes in volume | Otherwise a longer quarter looks like waste |
| Emission factor from data pack | Read from the data card; 0.4 kg/kWh is our assumption | The documented schema has no such file |
| Action costs | Assumed range in `estimate_savings` | Not in the data; payback is only as good as this assumption |

## 4. Remaining work, in order

1. `git init`, first commit, private repository from the organiser template.
2. Run once with a real key (`LLM_PROVIDER=anthropic`) and check the reworded brief passes the number check.
3. Start Docker Desktop and run `docker compose up --build`; fix whatever the first build shows.
4. Evaluation report for the agent path (answer-type accuracy, faithfulness rate, p50/p95 latency, tokens).
5. README, three sample traces, data card text, disclosures.

## 5. Known limits of the current evaluation

- The generator and the baseline share the same linear form, so errors on seeds 1-3 are optimistic. The stress set (curved weather response) is the more honest number.
- Intraday load shape is identical in both quarters, so load shifting is untested.
- One leak pattern is planted; other waste patterns are not covered.
- Nothing here has been run on the organisers' data.
