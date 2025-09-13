from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


def _add_src_to_path() -> None:
    import sys

    this = Path(__file__).resolve()
    root = this.parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

from paper_analysis.utils import ensure_dir  # noqa: E402


def run(cfg: Dict[str, Any]) -> None:
    import numpy as np
    import pandas as pd

    np.random.seed(int(cfg.get("parameters", {}).get("random_seed", 42)))

    paths = cfg.get("paths", {})
    raw_dir = Path(paths.get("raw", "data/raw"))
    processed_dir = ensure_dir(paths.get("processed", "data/processed"))

    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {raw_dir}. Creating a small dummy dataset …")
        df = pd.DataFrame(
            {
                "group": ["A"] * 50 + ["B"] * 50,
                "value": np.r_[np.random.normal(0, 1, 50), np.random.normal(1, 1, 50)],
            }
        )
        out = processed_dir / "dummy_processed.csv"
        df.to_csv(out, index=False)
        print(f"Wrote {out}")
        return

    for p in csvs:
        print(f"Processing {p.name} …")
        df = pd.read_csv(p)

        # Example cleaning (customize as needed)
        df = df.dropna()  # simple example: drop missing rows

        out = processed_dir / f"{p.stem}_processed.csv"
        df.to_csv(out, index=False)
        print(f"Wrote {out}")

