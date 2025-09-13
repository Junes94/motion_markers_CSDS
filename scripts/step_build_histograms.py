from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


def _compute_bin_edges(values: np.ndarray, method: str, manual_width: float | None = None) -> np.ndarray:
    data = values[np.isfinite(values)]
    if data.size == 0:
        return np.array([0.0, 1.0])
    x_min = np.percentile(data, 1)
    x_max = np.percentile(data, 99)
    if method == "sturges":
        n = int(np.ceil(np.log2(max(data.size, 1))) + 1)
    elif method == "scott":
        sigma = np.std(data)
        if sigma == 0:
            n = 10
        else:
            h = 3.5 * sigma * (data.size ** (-1 / 3))
            n = int(np.ceil((x_max - x_min) / max(h, 1e-6)))
    elif method == "freedman_diaconis":
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            n = 10
        else:
            h = 2 * iqr * (data.size ** (-1 / 3))
            n = int(np.ceil((x_max - x_min) / max(h, 1e-6)))
    elif method == "manual" and manual_width:
        n = int(np.ceil((x_max - x_min) / manual_width))
    else:
        n = 10
    n = max(n, 1)
    return np.linspace(x_min, x_max, n + 1)


def run(cfg: Dict[str, Any]) -> None:
    params = cfg.get("parameters", {})
    paths = cfg.get("paths", {})

    root = Path(__file__).resolve().parents[1]

    results_dir = Path(paths.get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = (root / results_dir).resolve()
    histogram_dir_path = Path(paths.get("histogram_dir", results_dir / "scalar_histograms"))
    if not histogram_dir_path.is_absolute():
        histogram_dir_path = (root / histogram_dir_path).resolve()
    histogram_dir = ensure_dir(histogram_dir_path)
    figs_dir_cfg = Path(paths.get("figures_dir", "results/figures"))
    if not figs_dir_cfg.is_absolute():
        figs_dir_cfg = (root / figs_dir_cfg).resolve()
    figs_dir = ensure_dir(figs_dir_cfg) if params.get("save_plots", False) else None

    # Allow overriding the scalars source via config paths.scalars_csv (e.g., ablation output)
    scalars_csv_cfg = paths.get("scalars_csv")
    if scalars_csv_cfg:
        scalars_csv = Path(scalars_csv_cfg)
        if not scalars_csv.is_absolute():
            scalars_csv = (root / scalars_csv).resolve()
    else:
        scalars_csv = results_dir / "scalar_summaries.csv"
    index_csv = Path(paths.get("group_index_csv", "data/SIT/SIratio.csv"))
    if not index_csv.is_absolute():
        index_csv = (root / index_csv).resolve()

    if not scalars_csv.exists():
        print(f"Scalar summary not found: {scalars_csv}")
        return
    if not index_csv.exists():
        print(f"Group index file not found: {index_csv}")
        return

    variables: List[str] = list(params.get("variables", []))
    bin_method: str = str(params.get("bin_method", "freedman_diaconis")).lower()
    manual_dist = float(params.get("manual_bin_width_distance_like", 1.0))
    manual_angle = float(params.get("manual_bin_width_angle_like", 0.5236))
    variable_bins = params.get("variable_bins", {})  # per-variable bins: int (count) or list (edges)

    df = pd.read_csv(scalars_csv)
    index_df = pd.read_csv(index_csv)
    if "group" in df.columns:
        df = df.drop(columns=["group"])  # avoid double merge
    merged = pd.merge(df, index_df, on="name", how="inner")

    groups = merged["group"].unique().tolist()
    if not variables:
        variables = [c for c in merged.columns if c not in {"name", "group"}]
    print(f"[build_histograms] results_dir={results_dir}")
    print(f"[build_histograms] index_csv={index_csv}")
    print(f"[build_histograms] variables={variables}")
    print(f"[build_histograms] bin_method={bin_method}")

    all_rows = []
    mouse_averages_rows = []
    for var in variables:
        values = merged[var].to_numpy()
        is_angle = var in {"angle_to_origin", "torso_angle"}
        width = manual_angle if is_angle else manual_dist
        # Per-variable override: integer => number of bins (1-99 pct range), list => explicit edges
        vb = variable_bins.get(var)
        if isinstance(vb, int) and vb > 0:
            data = values[np.isfinite(values)]
            if data.size == 0:
                edges = np.array([0.0, 1.0])
            else:
                x_min = np.percentile(data, 1)
                x_max = np.percentile(data, 99)
                edges = np.linspace(x_min, x_max, int(vb) + 1)
        elif isinstance(vb, (list, tuple, np.ndarray)) and len(vb) >= 2:
            edges = np.asarray(vb, dtype=float)
        else:
            edges = _compute_bin_edges(values, bin_method, manual_width=width)
        centers = (edges[:-1] + edges[1:]) / 2

        group_mouse_hist = {g: {} for g in groups}
        for name, sub in merged.groupby("name"):
            hist, _ = np.histogram(sub[var].to_numpy(), bins=edges)
            norm = hist / max(hist.sum(), 1)
            g = sub["group"].iloc[0]
            group_mouse_hist[g][name] = {"bin_centers": centers, "hist_normalized": norm}
            for c, v in zip(centers, norm):
                all_rows.append(
                    {
                        "variable": var,
                        "group": g,
                        "mouse": name,
                        "bin_center": float(c),
                        "normalized_frequency": float(v),
                    }
                )

            # Mouse-level summary statistics for this variable
            vals = sub[var].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                mouse_averages_rows.append(
                    {
                        "variable": var,
                        "group": g,
                        "name": name,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                        "n": int(vals.size),
                    }
                )

        # Save per-variable histogram CSV
        var_df = pd.DataFrame([r for r in all_rows if r["variable"] == var])
        out_csv = Path(histogram_dir) / f"{var}_histogram_data.csv"
        var_df.to_csv(out_csv, index=False)

    # Save combined group means
    all_df = pd.DataFrame(all_rows)
    if not all_df.empty:
        group_mean = (
            all_df.groupby(["variable", "group", "bin_center"])['normalized_frequency'].mean().reset_index()
        )
        group_mean_out = Path(histogram_dir) / "group_mean_histogram.csv"
        group_mean.to_csv(group_mean_out, index=False)
        print(f"Wrote {group_mean_out}")
    else:
        print("No histogram rows produced.")

    # Save per-mouse averages across variables
    if mouse_averages_rows:
        ma_df = pd.DataFrame(mouse_averages_rows)
        ma_out = Path(histogram_dir) / "group_mouse_averages_all.csv"
        ma_df.to_csv(ma_out, index=False)
        print(f"Wrote {ma_out}")
