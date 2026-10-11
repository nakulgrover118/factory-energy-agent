"""Load a folder of CSVs (Annex B schema) and print the validation output.

If the API is running, the files are sent to it (POST /ingest), so the running service sees them
at once. Otherwise they are written straight into the store.

Usage: python -m energy.loader data/seed_1 [--late-drop] [--api http://localhost:8000] [--store store]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from .pipeline import SCHEMA


def _print_checks(checks: list[dict]) -> None:
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FLAG'}] {c['id']}. {c['check']}: {c['detail']}")


def via_api(api: str, folder: Path, late_drop: bool) -> bool:
    try:
        requests.get(f"{api}/health", timeout=30).raise_for_status()  # the first call pings the LLM
    except requests.RequestException:
        return False
    # same machine or container: let the API read the folder itself (keeps the data card)
    r = requests.post(f"{api}/admin/load", json={"folder": str(folder.resolve()), "late_drop": late_drop}, timeout=600)
    if r.ok:
        body = r.json()
        print(f"Loaded {folder} through the running API at {api}")
        for name, x in body["loaded"].items():
            print(f"{name}: {x['status']}, loaded {x['rows_loaded']:,}, rejected {x['rows_rejected']:,}" + (f" ({x['how']})" if x.get("how") else ""))
        _print_checks(body["validation"])
        return True
    print(f"Sending files to the running API at {api}")
    out = {}
    files = [(name, folder / f"{name}.csv", "replace") for name in SCHEMA]
    if late_drop:
        files += [(name, folder / "late_drop" / f"{name}.csv", "auto") for name in SCHEMA]
    for name, path, mode in files:
        if path.exists():
            with open(path, "rb") as f:
                out = requests.post(f"{api}/ingest", files={"file": (path.name, f, "text/csv")},
                                    data={"meta": json.dumps({"table": name, "mode": mode})}, timeout=300).json()
            print(f"{path.relative_to(folder)}: {out['status']}, loaded {out['rows_loaded']:,}, rejected {out['rows_rejected']:,}"
                  + (f" ({out['how']})" if out.get("how") else "") + (f" - {out['detail']}" if out.get("detail") else ""))
    if out.get("validation"):
        _print_checks(out["validation"])
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--late-drop", action="store_true", help="apply the folder's late_drop files after loading")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--store", default=None)
    args = ap.parse_args()
    folder = Path(args.folder)
    if args.store is None and via_api(args.api.rstrip("/"), folder, args.late_drop):
        return
    import pandas as pd

    from .store import Workspace

    ws = Workspace(args.store)
    for table, r in ws.load_folder(folder).items():
        print(f"{table}: loaded {r['rows_loaded']:,}, rejected {r['rows_rejected']:,} ({r['status']})")
    if args.late_drop:
        for name in SCHEMA:
            f = folder / "late_drop" / f"{name}.csv"
            if f.exists():
                r = ws.ingest(name, pd.read_csv(f))
                print(f"late_drop/{name}: {r['status']}, loaded {r['rows_loaded']:,} ({r.get('how', '')})")
    if not ws.ready:
        print("Not all four tables are loaded; analysis is unavailable.")
        return
    _print_checks(ws.checks())
    print(json.dumps(ws.cleaned()[1], indent=2))


if __name__ == "__main__":
    main()
