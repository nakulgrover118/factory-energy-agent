"""Workspace: the loaded tables, the cleaned data derived from them, and the run trace.

Raw tables live as CSV in STORE_DIR so the API and the loader CLI share them.
Traces live in SQLite (STORE_DIR/trace.db) and never leave the machine.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd

from . import analytics as an
from .pipeline import SCHEMA, apply_ingest, clean, parse_table, validate

BASELINE_MODEL = os.getenv("BASELINE_MODEL", "ols")


class Workspace:
    def __init__(self, store_dir: str | Path | None = None):
        self.dir = Path(store_dir or os.getenv("STORE_DIR", "store"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.frames: dict = {}
        self.version = 0
        self._derived: dict = {}
        self._db = sqlite3.connect(self.dir / "trace.db", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, created REAL, record TEXT)")
        self._db.commit()
        self.reload()

    # ---- raw tables -------------------------------------------------------
    def reload(self) -> None:
        with self.lock:
            self.frames = {"_rejected": {}}
            for name in SCHEMA:
                f = self.dir / f"{name}.csv"
                if f.exists():
                    self.frames[name], self.frames["_rejected"][name] = parse_table(name, pd.read_csv(f))
            card = self.dir / "data_card.json"
            self.frames["_card"] = json.loads(card.read_text()) if card.exists() else None
            self._changed()

    def load_folder(self, folder: str | Path) -> dict:
        """Replace the workspace with the four CSVs in a folder (used by the loader CLI)."""
        folder = Path(folder)
        with self.lock:
            report = {}
            for name in SCHEMA:
                src = folder / f"{name}.csv"
                if src.exists():
                    report[name] = self.ingest(name, pd.read_csv(src), replace=True)
            if (folder / "data_card.json").exists():
                shutil.copy(folder / "data_card.json", self.dir / "data_card.json")
                self.frames["_card"] = json.loads((self.dir / "data_card.json").read_text())
            self._changed()
            return report

    def ingest(self, table: str, raw: pd.DataFrame, replace: bool = False) -> dict:
        if table not in SCHEMA:
            raise ValueError(f"unknown table '{table}'; expected one of {list(SCHEMA)}")
        new, rejected = parse_table(table, raw)
        with self.lock:
            old = self.frames.get(table)
            merged = new if (replace or old is None) else apply_ingest(table, old, new)
            self.frames[table] = merged
            self.frames.setdefault("_rejected", {})[table] = rejected
            merged.to_csv(self.dir / f"{table}.csv", index=False)
            self._changed()
        return {"rows_loaded": int(len(new)), "rows_rejected": int(rejected),
                "status": "loaded" if len(new) else "rejected"}

    def _changed(self) -> None:
        self.version += 1
        self._derived = {}

    @property
    def ready(self) -> bool:
        return all(name in self.frames and len(self.frames[name]) for name in SCHEMA)

    # ---- derived data (cached until the tables change) -----------------------
    def _get(self, key, build):
        with self.lock:
            if key not in self._derived:
                self._derived[key] = build()
            return self._derived[key]

    def checks(self) -> list[dict]:
        return self._get("checks", lambda: validate(self.frames))

    def cleaned(self) -> tuple[pd.DataFrame, dict]:
        return self._get("clean", lambda: clean(self.frames))

    def quarters(self) -> list[str]:
        """Quarters with at least 30 days of meter data, oldest first."""
        idx = self.cleaned()[0].index
        days = pd.Series(1, index=idx.normalize().unique()).groupby(lambda d: str(d.to_period("Q"))).sum()
        return [f"{q[:4]}-{q[4:]}" for q in days[days >= 30].index]

    def daily(self, compare_to: str) -> pd.DataFrame:
        return self._get(("daily", compare_to), lambda: an.daily_table(self.cleaned()[0], self.frames, compare_to))

    def bridge(self, period: str, compare_to: str) -> dict:
        return self._get(("bridge", period, compare_to),
                         lambda: an.decompose(self.daily(compare_to), period, compare_to, model=BASELINE_MODEL))

    def emission_factor(self) -> tuple[float, str]:
        card = self.frames.get("_card") or {}
        if "emission_factor_kg_per_kwh" in card:
            return float(card["emission_factor_kg_per_kwh"]), "data card"
        return float(os.getenv("EMISSION_FACTOR", "0.4")), "assumed default (not in the loaded data)"

    # ---- trace --------------------------------------------------------------
    def save_trace(self, record: dict) -> None:
        with self.lock:
            self._db.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?)",
                             (record["run_id"], time.time(), json.dumps(record, default=str)))
            self._db.commit()

    def get_trace(self, run_id: str) -> dict | None:
        row = self._db.execute("SELECT record FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None
