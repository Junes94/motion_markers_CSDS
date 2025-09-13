from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _add_src_to_path() -> None:
    import sys

    this = Path(__file__).resolve()
    root = this.parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

from paper_analysis.io import load_pose_folder  # noqa: E402
from paper_analysis.utils import ensure_dir  # noqa: E402
from paper_analysis.features import compute_scalar_summary  # noqa: E402


def run(cfg: Dict[str, Any]) -> None:
    params = cfg.get("parameters", {})
    paths = cfg.get("paths", {})

    root = Path(__file__).resolve().parents[1]

    pose_dir = Path(paths.get("pose_dir", "data/pose_traj"))
    if not pose_dir.is_absolute():
        pose_dir = (root / pose_dir).resolve()

    results_dir = Path(paths.get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = (root / results_dir).resolve()
    ensure_dir(results_dir)

    # Read runtime parameters from config (defaults apply only if missing)
    fps = int(params.get("fps", 30))
    smoothing_window = params.get("smoothing_window", 5)
    origin = tuple(params.get("origin", [0.0, 0.0]))  # type: ignore
    centerpoint = tuple(params.get("centerpoint", ["head", "torso"]))  # type: ignore
    length_criteria = tuple(params.get("length_criteria", ["head", "anus"]))  # type: ignore
    height_criteria = tuple(params.get("height_criteria", ["head", "anus"]))  # type: ignore
    velocity_criteria = tuple(params.get("velocity_criteria", ["head", "torso"]))  # type: ignore

    # Labels can be provided directly or via a file. If omitted, original CSV headers are used.
    labels = params.get("labels")
    labels_file = params.get("labels_file")
    if labels is None and labels_file:
        lf = Path(labels_file)
        if not lf.is_absolute():
            lf = (root / labels_file).resolve()
        if lf.exists():
            labels = [line.strip() for line in lf.read_text(encoding="utf-8").splitlines() if line.strip()]

    exclude_keypoints = params.get("exclude_keypoints", [])
    coord_suffixes = params.get("coord_suffixes", ["_x", "_y", "_z"])
    pose_has_header = params.get("pose_has_header", False)

    # Log effective settings for transparency
    print(f"[compute_scalars] pose_dir={pose_dir}")
    print(f"[compute_scalars] fps={fps}, smoothing_window={smoothing_window}, origin={origin}")

    dfs, names = load_pose_folder(
        str(pose_dir),
        labels=labels,
        exclude_keypoints=exclude_keypoints,
        coord_suffixes=coord_suffixes,
        has_header=bool(pose_has_header),
    )

    rows: List[pd.DataFrame] = []
    for df, name in zip(dfs, names):
        scalars = compute_scalar_summary(
            df,
            fps=fps,
            origin=(float(origin[0]), float(origin[1])),
            smoothing_window=int(smoothing_window) if smoothing_window else None,
            centerpoint=centerpoint,
            length_criteria=length_criteria,
            height_criteria=height_criteria,
            velocity_criteria=velocity_criteria,
        )
        scalars["name"] = name
        rows.append(scalars)

    if not rows:
        print(f"No CSV files found in {pose_dir}.")
        return

    out = pd.concat(rows, ignore_index=True)
    out_path = Path(results_dir) / "scalar_summaries.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
