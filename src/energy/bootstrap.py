"""Optional first-start step for the container: if the store is empty and SEED_DEMO_DATA=true,
generate one synthetic dataset and load it, so `docker compose up` shows a working system at once.
The sample stays in STORE_DIR/samples, so its late data drop can be applied from the UI.
Data loaded later through /ingest, /admin/load or the loader replaces it."""
from __future__ import annotations

import os

from .generate import generate
from .settings import load_env
from .store import Workspace


def main() -> None:
    load_env()
    ws = Workspace()
    if ws.ready or os.getenv("SEED_DEMO_DATA", "").lower() not in ("1", "true", "yes"):
        print(f"bootstrap: store {'ready' if ws.ready else 'empty'}, nothing generated")
        return
    seed = int(os.getenv("DEMO_SEED", "1"))
    folder = ws.dir / "samples" / f"seed_{seed}"
    generate(seed, folder)
    ws.load_folder(folder)
    print(f"bootstrap: loaded synthetic demo data (seed {seed}) from {folder}")


if __name__ == "__main__":
    main()
