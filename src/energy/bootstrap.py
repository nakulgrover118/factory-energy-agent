"""Optional first-start step for the container: if the store is empty and SEED_DEMO_DATA=true,
generate one synthetic dataset and load it, so `docker compose up` shows a working system.
Data loaded later through /ingest or the loader replaces it."""
from __future__ import annotations

import os
import tempfile

from .generate import generate
from .store import Workspace


def main() -> None:
    ws = Workspace()
    if ws.ready or os.getenv("SEED_DEMO_DATA", "").lower() not in ("1", "true", "yes"):
        print(f"bootstrap: store {'ready' if ws.ready else 'empty'}, nothing generated")
        return
    with tempfile.TemporaryDirectory() as tmp:
        generate(int(os.getenv("DEMO_SEED", "1")), tmp)
        ws.load_folder(tmp)
    print("bootstrap: loaded synthetic demo data (seed %s)" % os.getenv("DEMO_SEED", "1"))


if __name__ == "__main__":
    main()
