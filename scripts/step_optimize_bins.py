from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _add_src_to_path() -> None:
    import sys

    this = Path(__file__).resolve()
    root = this.parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

from paper_analysis.utils import ensure_dir  # noqa: E402
from paper_analysis.features import freedman_diaconis_bins, scott_bins, sturges_bins  # noqa: E402


def run(cfg: Dict[str, Any]) -> None:
    params = cfg.get("parameters", {})
    paths = cfg.get("paths", {})

    variables: List[str] = list(params.get("variables", []))

    results_dir = Path(paths.get("results_dir", "results"))
    scalars_csv = results_dir / "scalar_summaries.csv"
    if not scalars_csv.exists():
        print(f"Scalar summary not found: {scalars_csv}")
        return

    bins_out_dir = ensure_dir(results_dir / "configs")
    out_csv = Path(bins_out_dir) / "histogram_bin_recommendations.csv"

    df = pd.read_csv(scalars_csv)
    if not variables:
        variables = [c for c in df.columns if c not in {"name", "group"}]

    recs = []
    for var in variables:
        # compute average recommended bins across mice
        per_mouse = []
        for name, sub in df.groupby("name"):
            arr = sub[var].to_numpy()
            per_mouse.append(
                {
                    "name": name,
                    "sturges": sturges_bins(arr),
                    "freedman_diaconis": freedman_diaconis_bins(arr),
                    "scott": scott_bins(arr),
                }
            )
        if not per_mouse:
            continue
        avg_sturges = float(np.mean([x["sturges"] for x in per_mouse]))
        avg_fd = float(np.mean([x["freedman_diaconis"] for x in per_mouse]))
        avg_scott = float(np.mean([x["scott"] for x in per_mouse]))
        recs.append(
            {
                "variable": var,
                "sturges": round(avg_sturges, 2),
                "freedman_diaconis": round(avg_fd, 2),
                "scott": round(avg_scott, 2),
            }
        )

    pd.DataFrame(recs).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

