"""Workspace: the loaded tables, the cleaned data derived from them, results, and the run trace.

Data lives in DuckDB (STORE_DIR/energy.duckdb) in three layers:
  raw_<table>    the four tables as loaded (Annex B schema)
  clean_meter    the gap-free 15-minute table, one column per meter
  ingest_log     every load, with what it changed
Run traces and manager approvals live in SQLite (STORE_DIR/trace.db).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import duckdb
import pandas as pd

from . import analytics as an
from .pipeline import MAX_ESTIMATED_SHARE, SCHEMA, apply_ingest, clean, parse_table, validate

BASELINE_MODEL = os.getenv("BASELINE_MODEL", "ols")


class Workspace:
    def __init__(self, store_dir: str | Path | None = None):
        self.dir = Path(store_dir or os.getenv("STORE_DIR", "store"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.frames: dict = {}
        self.version = 0
        self._derived: dict = {}
        self.last_change: dict | None = None
        self.duck = duckdb.connect(str(self.dir / "energy.duckdb"))
        self.duck.execute("CREATE TABLE IF NOT EXISTS ingest_log (loaded_at DOUBLE, version INTEGER, tables VARCHAR, "
                          "rows_loaded INTEGER, rows_rejected INTEGER, how VARCHAR)")
        self.duck.execute("CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
        self._db = sqlite3.connect(self.dir / "trace.db", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, created REAL, record TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS approvals (action_key TEXT PRIMARY KEY, decision TEXT, note TEXT, "
                         "decided_at REAL, data_version INTEGER)")
        self._db.commit()
        self.reload()

    # ---- raw tables --------------------------------------------------------------
    def _meta(self, key: str, value=None):
        if value is None:
            row = self.duck.execute("SELECT value FROM meta WHERE key = ?", [key]).fetchone()
            return json.loads(row[0]) if row else None
        self.duck.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", [key, json.dumps(value, default=str)])

    def _tables(self) -> set[str]:
        return {r[0] for r in self.duck.execute("SELECT table_name FROM information_schema.tables").fetchall()}

    def reload(self) -> None:
        with self.lock:
            self.frames = {"_rejected": {}}
            existing = self._tables()
            for name in SCHEMA:
                if f"raw_{name}" in existing:
                    self.frames[name] = self.duck.execute(f"SELECT * FROM raw_{name}").df()
                    self.frames["_rejected"][name] = 0
            self.frames["_card"] = self._meta("data_card")
            self.frames["_card_stale"] = bool(self._meta("card_stale"))
            self.version = int(self._meta("version") or 0)
            self._derived = {}

    def _save(self, name: str, df: pd.DataFrame) -> None:
        self.duck.register("_incoming", df)
        self.duck.execute(f"CREATE OR REPLACE TABLE raw_{name} AS SELECT * FROM _incoming")
        self.duck.unregister("_incoming")

    def _snapshot(self) -> dict | None:
        """The headline numbers before a change, so the next answer can say what moved."""
        if not self.ready:
            return None
        try:
            return self.summary()
        except Exception:
            return None

    def _changed(self, tables: list[str], rows: int, rejected: int, how: str, before: dict | None) -> None:
        self.version += 1
        self._meta("version", self.version)
        self._derived = {}
        self.duck.execute("INSERT INTO ingest_log VALUES (?, ?, ?, ?, ?, ?)", [time.time(), self.version, ",".join(tables), rows, rejected, how])
        prev = self.last_change
        if prev and prev.get("before") and tables and prev["tables"] and before is not None:
            # consecutive updates (a late drop is often several files) are compared with the state before the first
            self.last_change = {"version": self.version, "tables": prev["tables"] + tables,
                                "how": prev["how"] + "; " + how, "rows_loaded": prev["rows_loaded"] + rows, "before": prev["before"]}
        else:
            self.last_change = {"version": self.version, "tables": tables, "how": how, "rows_loaded": rows, "before": before}

    def load_folder(self, folder: str | Path) -> dict:
        """Replace the workspace with the four CSVs in a folder (the loader CLI and /admin/load)."""
        folder = Path(folder)
        with self.lock:
            before = self._snapshot()
            report, total, rejected_total = {}, 0, 0
            for name in SCHEMA:
                src = folder / f"{name}.csv"
                if src.exists():
                    new, rejected = parse_table(name, pd.read_csv(src))
                    self.frames[name] = new
                    self.frames["_rejected"][name] = rejected
                    self._save(name, new)
                    report[name] = {"rows_loaded": int(len(new)), "rows_rejected": int(rejected),
                                    "status": "loaded" if len(new) else "rejected"}
                    total, rejected_total = total + len(new), rejected_total + rejected
            card_file = folder / "data_card.json"
            card = json.loads(card_file.read_text()) if card_file.exists() else None
            self.frames["_card"], self.frames["_card_stale"] = card, False
            self._meta("data_card", card or {})
            self._meta("card_stale", False)
            self._meta("last_folder", str(folder.resolve()))
            self.last_change = None
            self._changed(list(report), total, rejected_total, f"loaded the folder {folder.name}", None)
            return report

    def ingest(self, table: str, raw: pd.DataFrame, mode: str = "auto") -> dict:
        if table not in SCHEMA:
            raise ValueError(f"unknown table '{table}'; expected one of {list(SCHEMA)}")
        new, rejected = parse_table(table, raw)
        if new.empty:
            return {"rows_loaded": 0, "rows_rejected": int(rejected), "status": "rejected", "detail": "no valid rows"}
        with self.lock:
            before = self._snapshot()
            merged, how = apply_ingest(table, self.frames.get(table), new, mode)
            self.frames[table] = merged
            self.frames["_rejected"][table] = rejected
            self._save(table, merged)
            if self.frames.get("_card"):
                self.frames["_card_stale"] = True
                self._meta("card_stale", True)
            self._changed([table], len(new), rejected, f"{table}: {how}", before)
        return {"rows_loaded": int(len(new)), "rows_rejected": int(rejected), "status": "loaded", "how": how}

    def reset(self) -> None:
        with self.lock:
            for name in SCHEMA:
                self.duck.execute(f"DROP TABLE IF EXISTS raw_{name}")
            self.duck.execute("DROP TABLE IF EXISTS clean_meter")
            self._meta("data_card", {})
            self.frames = {"_rejected": {}, "_card": None, "_card_stale": False}
            self._changed([], 0, 0, "reset: all data removed", None)

    @property
    def ready(self) -> bool:
        return all(name in self.frames and len(self.frames[name]) for name in SCHEMA)

    # ---- derived data (cached until the tables change) ---------------------------------
    def _get(self, key, build):
        with self.lock:
            if key not in self._derived:
                self._derived[key] = build()
            return self._derived[key]

    def checks(self) -> list[dict]:
        return self._get("checks", lambda: validate(self.frames))

    def cleaned(self) -> tuple[pd.DataFrame, dict]:
        def build():
            meters, report = clean(self.frames)
            self.duck.register("_clean", meters.reset_index())
            self.duck.execute("CREATE OR REPLACE TABLE clean_meter AS SELECT * FROM _clean")
            self.duck.unregister("_clean")
            return meters, report
        return self._get("clean", build)

    def sql(self, query: str, params: list | None = None) -> pd.DataFrame:
        """Query the validated data (the clean_meter and raw_* tables) with DuckDB SQL."""
        self.cleaned()
        with self.lock:
            return self.duck.execute(query, params or []).df()

    def blocking_issue(self) -> str | None:
        """A data problem serious enough that no cost figure should be given."""
        share = self.cleaned()[1].get("estimated_share_by_meter", {}).get("main", 0.0)
        if share > 4 * MAX_ESTIMATED_SHARE:
            return f"{round(100 * share)}% of the main meter's intervals are missing, so the bill cannot be reconstructed"
        return None

    def quarters(self) -> list[str]:
        """Quarters with at least 30 days of meter data, oldest first."""
        idx = self.cleaned()[0].index
        days = pd.Series(1, index=idx.normalize().unique()).groupby(lambda d: str(d.to_period("Q"))).sum()
        return [f"{q[:4]}-{q[4:]}" for q in days[days >= 30].index]

    @staticmethod
    def period_bounds(period: str) -> tuple[str, str]:
        p = pd.Period(period.replace("-", ""), freq="Q")
        return p.start_time.strftime("%Y-%m-%d %H:%M:%S"), (p.end_time + pd.Timedelta(1, "ns")).strftime("%Y-%m-%d %H:%M:%S")

    def default_periods(self) -> tuple[str | None, str | None]:
        q = self.quarters()
        return (q[-1] if q else None), (q[-2] if len(q) > 1 else None)

    def daily(self, compare_to: str) -> pd.DataFrame:
        return self._get(("daily", compare_to), lambda: an.daily_table(self.cleaned()[0], self.frames, compare_to))

    def bridge(self, period: str, compare_to: str) -> dict:
        return self._get(("bridge", period, compare_to),
                         lambda: an.decompose(self.daily(compare_to), period, compare_to, model=BASELINE_MODEL))

    def leaks(self, period: str, compare_to: str) -> list[dict]:
        return self._get(("leaks", period, compare_to), lambda: an.detect_leaks(self.cleaned()[0], self.frames, period, compare_to))

    def findings(self, period: str, compare_to: str) -> list[dict]:
        return self._get(("findings", period, compare_to),
                         lambda: an.off_hours_waste(self.cleaned()[0], self.frames, period, self.leaks(period, compare_to)))

    def actions(self, period: str, compare_to: str) -> list[dict]:
        def build():
            factor, _ = self.emission_factor()
            acts = an.estimate_savings(self.findings(period, compare_to), self.leaks(period, compare_to),
                                       self.cleaned()[0], self.frames, factor)
            decided = self.approvals()
            for a in acts:
                a["action_key"] = f"{a['asset']}|{a['window']}"
                a["approval"] = decided.get(a["action_key"], {"decision": "pending"})
            return acts
        return self._get(("actions", period, compare_to), build)

    def emission_factor(self) -> tuple[float, str]:
        card = self.frames.get("_card") or {}
        if "emission_factor_kg_per_kwh" in card:
            return float(card["emission_factor_kg_per_kwh"]), "data card"
        return float(os.getenv("EMISSION_FACTOR", "0.71")), "assumed default (not in the loaded data)"

    def currency(self) -> str:
        return (self.frames.get("_card") or {}).get("currency") or os.getenv("CURRENCY", "INR")

    def summary(self, period: str | None = None, compare_to: str | None = None) -> dict:
        """Headline numbers for the dashboard and for the before/after comparison."""
        p0, c0 = self.default_periods()
        period, compare_to = period or p0, compare_to or c0
        b = self.bridge(period, compare_to)
        acts = self.actions(period, compare_to)
        return {"period": period, "compare_to": compare_to, "status": b.get("status"),
                "total_change_pct": b.get("total_change_pct"), "bill_reference": b.get("bill_reference"),
                "bill_current": b.get("bill_current"),
                "bridge": {x["cause"]: x["pct_points"] for x in b.get("bridge", [])},
                "savings": an.savings_summary(acts), "actions": [a["action"] for a in acts], "data_version": self.version,
                "assets": {r["asset"]: r["kwh"] for r in an.asset_breakdown(self.cleaned()[0], period)}}

    def changes(self) -> dict | None:
        """What moved since the last data load: the cause points before and after."""
        ch = self.last_change
        if not ch or not ch.get("before") or not self.ready:
            return None
        before = ch["before"]
        after = self.summary(before["period"], before["compare_to"])
        if after.get("status") != "ok" or before.get("status") != "ok":
            return None
        delta = {c: round(after["bridge"][c] - before["bridge"].get(c, 0.0), 2) for c in after["bridge"]}
        assets = {a: round(100 * (after["assets"][a] - before["assets"][a]) / before["assets"][a], 1)
                  for a in after["assets"] if a in before.get("assets", {}) and before["assets"][a]}
        assets = {a: v for a, v in assets.items() if abs(v) >= 0.5}
        return {"change": ch["how"], "tables": ch["tables"], "version": ch["version"],
                "total_before": before["total_change_pct"], "total_after": after["total_change_pct"],
                "bill_before": before["bill_current"], "bill_after": after["bill_current"],
                "cause_before": before["bridge"], "cause_after": after["bridge"], "cause_delta": delta,
                "savings_before": before["savings"]["saving_money_per_year"], "savings_after": after["savings"]["saving_money_per_year"],
                "asset_kwh_change_pct": assets}

    def ingest_log(self, limit: int = 10) -> list[dict]:
        df = self.duck.execute("SELECT * FROM ingest_log ORDER BY loaded_at DESC LIMIT ?", [limit]).df()
        return df.to_dict("records")

    # ---- approvals and trace ----------------------------------------------------------
    def approve(self, action_key: str, decision: str, note: str = "") -> dict:
        with self.lock:
            self._db.execute("INSERT OR REPLACE INTO approvals VALUES (?, ?, ?, ?, ?)",
                             (action_key, decision, note, time.time(), self.version))
            self._db.commit()
            self._derived = {k: v for k, v in self._derived.items() if not (isinstance(k, tuple) and k[0] == "actions")}
        return {"action_key": action_key, "decision": decision, "note": note}

    def approvals(self) -> dict:
        rows = self._db.execute("SELECT action_key, decision, note, decided_at FROM approvals").fetchall()
        return {r[0]: {"decision": r[1], "note": r[2], "decided_at": r[3]} for r in rows}

    def save_trace(self, record: dict) -> None:
        with self.lock:
            self._db.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?)",
                             (record["run_id"], time.time(), json.dumps(record, default=str)))
            self._db.commit()

    def get_trace(self, run_id: str) -> dict | None:
        row = self._db.execute("SELECT record FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None
