"""Load a folder of CSVs (Annex B schema) into the store and print the validation output.

Usage: python -m energy.loader data/seed_1 [--store store]
"""
from __future__ import annotations

import argparse
import json

from .store import Workspace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--store", default=None)
    args = ap.parse_args()
    ws = Workspace(args.store)
    for table, r in ws.load_folder(args.folder).items():
        print(f"{table}: loaded {r['rows_loaded']}, rejected {r['rows_rejected']} ({r['status']})")
    if not ws.ready:
        print("Not all four tables are loaded; analysis is unavailable.")
        return
    for c in ws.checks():
        print(f"[{'PASS' if c['passed'] else 'FLAG'}] {c['id']}. {c['check']}: {c['detail']}")
    print(json.dumps(ws.cleaned()[1], indent=2))


if __name__ == "__main__":
    main()
