from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_ANIMAL_SIZE_CFG: Dict[str, Any] = {
    "enabled": True,
    "output_csv": "animal_size_summary.csv",
    "instances": {
        "enabled": True,
        "max_instances_per_recording": 20,
        "selection": "first",
        "random_seed": 42,
        "walk_output_csv": "animal_size_walk_instances.csv",
        "rearing_output_csv": "animal_size_rearing_instances.csv",
    },
    "rearing": {
        "center_reference": "torso",
        "center_keypoints": ["head", "torso"],
        "hindlimb_keypoints": ["RH", "LH"],
        "baseline_percentile": 25,
        "height_factor": 2.0,
        "min_center_rise": 1.5,
        "min_hind_height": 2.0,
        "median_window_seconds": 0.25,
        "min_duration_seconds": 0.5,
    },
    "walk": {
        "head_keypoint": "head",
        "torso_keypoint": "torso",
        "velocity_threshold": 0.1,
        "angle_threshold_deg": 90,
        "bout_distance_threshold": 5.0,
    },
}


def _merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _add_src_to_path() -> None:
    import sys

    this = Path(__file__).resolve()
    root = this.parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

from src.features import compute_scalar_summary  # noqa: E402
from src.io import load_pose_folder  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


def _ensure_odd_window(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    if value % 2 == 0:
        value += 1
    return value


def _contiguous_true_segments(bool_series: pd.Series) -> List[Tuple[int, int]]:
    arr = bool_series.fillna(False).to_numpy(dtype=bool)
    if arr.size == 0:
        return []

    segments: List[Tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(arr):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, arr.size - 1))
    return segments


def _sample_bout_segments(
    segments: List[Tuple[int, int]],
    max_instances: int,
    selection: str = "first",
    random_seed: int = 0,
) -> List[Tuple[int, int]]:
    if max_instances <= 0:
        return []
    if len(segments) <= max_instances:
        return segments
    if selection == "random":
        rng = np.random.RandomState(random_seed)
        idx = sorted(rng.choice(len(segments), size=max_instances, replace=False).tolist())
        return [segments[i] for i in idx]
    return segments[:max_instances]


def _export_behavior_sampled_instances_csv(
    segments_by_name: Dict[str, List[Tuple[int, int]]],
    out_path: Path,
    *,
    max_instances_per_recording: int,
    selection: str,
    random_seed: int,
) -> None:
    columns: Dict[str, pd.Series] = {}
    for i, name in enumerate(sorted(segments_by_name.keys())):
        sampled = _sample_bout_segments(
            segments=segments_by_name[name],
            max_instances=max_instances_per_recording,
            selection=selection,
            random_seed=random_seed + i,
        )
        tuples_as_text = [f"('{name}', {start}, {end})" for start, end in sampled]
        columns[name] = pd.Series(tuples_as_text)

    pd.DataFrame(columns).to_csv(out_path, index=False, encoding="utf-8-sig")


def _angle_between_2d_series(v1: pd.DataFrame, v2: pd.DataFrame) -> pd.Series:
    a = v1.to_numpy(dtype=float)
    b = v2.to_numpy(dtype=float)
    dot = np.einsum("ij,ij->i", a, b)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.maximum(na * nb, 1e-9)
    cos = np.clip(dot / denom, -1.0, 1.0)
    return pd.Series(np.degrees(np.arccos(cos)), index=v1.index)


def _resolve_center_z(data: pd.DataFrame, center_reference: str, center_keypoints: Sequence[str]) -> pd.Series:
    ref = str(center_reference).lower().strip()
    if ref == "torso":
        return data["torso_z"]
    if ref == "head":
        return data["head_z"]
    if ref == "midpoint":
        if len(center_keypoints) != 2:
            raise ValueError("animal_size.rearing.center_keypoints must have exactly 2 keypoints")
        k1, k2 = center_keypoints[0], center_keypoints[1]
        return (data[f"{k1}_z"] + data[f"{k2}_z"]) / 2.0
    raise ValueError("animal_size.rearing.center_reference must be one of [midpoint, torso, head]")


def _rearing_bool(data: pd.DataFrame, fps: float, animal_cfg: Dict[str, Any]) -> pd.Series:
    rear = animal_cfg["rearing"]
    center_reference = str(rear.get("center_reference", "torso"))
    center_keypoints = rear.get("center_keypoints", ["head", "torso"])
    hindlimbs = rear.get("hindlimb_keypoints", ["RH", "LH"])
    if len(hindlimbs) != 2:
        raise ValueError("animal_size.rearing.hindlimb_keypoints must have exactly 2 keypoints")

    baseline_percentile = float(rear.get("baseline_percentile", 25.0))
    height_factor = float(rear.get("height_factor", 2.0))
    min_center_rise = float(rear.get("min_center_rise", 1.5))
    min_hind_height = float(rear.get("min_hind_height", 2.0))
    median_window_seconds = float(rear.get("median_window_seconds", 0.25))
    min_duration_seconds = float(rear.get("min_duration_seconds", 0.5))

    center_z = _resolve_center_z(data, center_reference, center_keypoints)
    h1_z = data[f"{hindlimbs[0]}_z"]
    h2_z = data[f"{hindlimbs[1]}_z"]

    baseline = float(np.percentile(center_z.dropna().to_numpy(dtype=float), baseline_percentile))
    threshold = max(baseline * height_factor, baseline + min_center_rise)
    raw = (center_z > threshold) & ((h1_z < min_hind_height) | (h2_z < min_hind_height))

    median_window = _ensure_odd_window(round(fps * median_window_seconds))
    min_duration_frames = max(1, int(round(fps * min_duration_seconds)))

    smoothed = raw.rolling(window=median_window, center=True, min_periods=1).median() == 1
    labels = smoothed.ne(smoothed.shift()).cumsum()
    durations = smoothed.groupby(labels).transform("size")
    final = smoothed & (durations >= min_duration_frames)
    return final.fillna(False).astype(bool)


def _walk_bool(data: pd.DataFrame, fps: float, rearing_bool: pd.Series, animal_cfg: Dict[str, Any]) -> pd.Series:
    walk = animal_cfg["walk"]
    head_kp = str(walk.get("head_keypoint", "head"))
    torso_kp = str(walk.get("torso_keypoint", "torso"))
    vel_threshold = float(walk.get("velocity_threshold", 0.1))
    angle_threshold_deg = float(walk.get("angle_threshold_deg", 90.0))
    bout_distance_threshold = float(walk.get("bout_distance_threshold", 5.0))

    head_2d = data[[f"{head_kp}_x", f"{head_kp}_y"]]
    torso_2d = data[[f"{torso_kp}_x", f"{torso_kp}_y"]]

    v_head = head_2d.diff(axis=0)
    v_torso = torso_2d.diff(axis=0)
    torso_to_head = pd.DataFrame(
        head_2d.to_numpy(dtype=float) - torso_2d.to_numpy(dtype=float),
        index=data.index,
        columns=["x", "y"],
    )

    speed_head = (v_head**2).sum(axis=1, skipna=False) ** 0.5
    speed_torso = (v_torso**2).sum(axis=1, skipna=False) ** 0.5
    crit_speed = (speed_head > vel_threshold) & (speed_torso > vel_threshold)

    ang_head = _angle_between_2d_series(torso_to_head, v_head)
    ang_torso = _angle_between_2d_series(torso_to_head, v_torso)
    crit_angle = (ang_head < angle_threshold_deg) & (ang_torso < angle_threshold_deg)

    walk123 = (crit_speed & crit_angle & (~rearing_bool)).fillna(False)
    out = walk123.copy()

    torso_speed = speed_torso.fillna(0.0)
    for start, end in _contiguous_true_segments(walk123):
        if float(torso_speed.iloc[start : end + 1].sum()) < bout_distance_threshold:
            out.iloc[start : end + 1] = False
    return out.astype(bool)


def _behavior_stats(values: pd.Series, bool_mask: pd.Series, prefix: str) -> Dict[str, float]:
    selected = values[bool_mask].dropna()
    return {
        f"{prefix}_mean": float(selected.mean()) if not selected.empty else np.nan,
        f"{prefix}_median": float(selected.median()) if not selected.empty else np.nan,
        f"{prefix}_frame_count": int(bool_mask.fillna(False).sum()),
    }


def run(cfg: Dict[str, Any]) -> None:
    params = cfg.get("parameters", {})
    animal_cfg = _merge_dict(DEFAULT_ANIMAL_SIZE_CFG, params.get("animal_size", {}))
    if not bool(animal_cfg.get("enabled", True)):
        print("[animal_size] disabled by config")
        return

    paths = cfg.get("paths", {})
    root = Path(__file__).resolve().parents[1]

    pose_dir = Path(paths.get("pose_dir", "data/pose_traj"))
    if not pose_dir.is_absolute():
        pose_dir = (root / pose_dir).resolve()

    results_dir = Path(paths.get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = (root / results_dir).resolve()

    animal_size_dir_cfg = paths.get("animal_size_dir")
    if animal_size_dir_cfg:
        out_dir = Path(animal_size_dir_cfg)
    else:
        out_dir = results_dir / "animal_size"
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    ensure_dir(out_dir)

    fps = float(params.get("fps", 20))
    smoothing_window = params.get("smoothing_window", 5)
    origin = tuple(params.get("origin", [0.0, 0.0]))
    centerpoint = tuple(params.get("centerpoint", ["torso", "torso"]))
    length_criteria = tuple(params.get("length_criteria", ["head", "anus"]))
    height_criteria = tuple(params.get("height_criteria", ["head", "anus"]))
    velocity_criteria = tuple(params.get("velocity_criteria", ["head", "torso"]))

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

    print(f"[animal_size] pose_dir={pose_dir}")
    print(f"[animal_size] output_dir={out_dir}")

    dfs, names = load_pose_folder(
        str(pose_dir),
        labels=labels,
        exclude_keypoints=exclude_keypoints,
        coord_suffixes=coord_suffixes,
        has_header=bool(pose_has_header),
    )

    if not dfs:
        print(f"[animal_size] no pose csv files found in {pose_dir}")
        return

    instances_cfg = animal_cfg["instances"]
    export_instances = bool(instances_cfg.get("enabled", True))
    max_instances_per_recording = int(instances_cfg.get("max_instances_per_recording", 20))
    selection = str(instances_cfg.get("selection", "first")).lower().strip()
    random_seed = int(instances_cfg.get("random_seed", 42))
    walk_instances_name = str(instances_cfg.get("walk_output_csv", "animal_size_walk_instances.csv"))
    rearing_instances_name = str(instances_cfg.get("rearing_output_csv", "animal_size_rearing_instances.csv"))
    if selection not in {"first", "random"}:
        raise ValueError("animal_size.instances.selection must be one of ['first', 'random']")

    rows: List[Dict[str, Any]] = []
    walk_segments_by_name: Dict[str, List[Tuple[int, int]]] = {}
    rearing_segments_by_name: Dict[str, List[Tuple[int, int]]] = {}
    for df, name in zip(dfs, names):
        try:
            scalars = compute_scalar_summary(
                df,
                fps=int(round(fps)),
                origin=(float(origin[0]), float(origin[1])),
                smoothing_window=int(smoothing_window) if smoothing_window else None,
                centerpoint=centerpoint,
                length_criteria=length_criteria,
                height_criteria=height_criteria,
                velocity_criteria=velocity_criteria,
            )

            rear_mask = _rearing_bool(df, fps=fps, animal_cfg=animal_cfg)
            walk_mask = _walk_bool(df, fps=fps, rearing_bool=rear_mask, animal_cfg=animal_cfg)
            walk_segments = _contiguous_true_segments(walk_mask)
            rear_segments = _contiguous_true_segments(rear_mask)
            walk_segments_by_name[name] = walk_segments
            rearing_segments_by_name[name] = rear_segments

            row: Dict[str, Any] = {"name": name, "total_frames": int(len(df))}
            row.update(_behavior_stats(scalars["length"], walk_mask, "walk_length"))
            row.update(_behavior_stats(scalars["height"], walk_mask, "walk_height"))
            row.update(_behavior_stats(scalars["length"], rear_mask, "rearing_length"))
            row.update(_behavior_stats(scalars["height"], rear_mask, "rearing_height"))
            row["walk_bout_count"] = int(len(walk_segments))
            row["rearing_bout_count"] = int(len(rear_segments))
            rows.append(row)
        except KeyError as e:
            print(f"[WARN] {name}: missing required column for animal_size ({e}). skipped")

    if not rows:
        print("[animal_size] no valid recordings for summary")
        return

    out_df = pd.DataFrame(rows).sort_values(by="name").reset_index(drop=True)
    out_name = str(animal_cfg.get("output_csv", "animal_size_summary.csv"))
    out_path = out_dir / out_name
    out_df.to_csv(out_path, index=False)
    print(f"[animal_size] wrote {out_path} ({len(out_df)} recordings)")

    if export_instances:
        walk_instances_path = out_dir / walk_instances_name
        rearing_instances_path = out_dir / rearing_instances_name
        _export_behavior_sampled_instances_csv(
            segments_by_name=walk_segments_by_name,
            out_path=walk_instances_path,
            max_instances_per_recording=max_instances_per_recording,
            selection=selection,
            random_seed=random_seed,
        )
        _export_behavior_sampled_instances_csv(
            segments_by_name=rearing_segments_by_name,
            out_path=rearing_instances_path,
            max_instances_per_recording=max_instances_per_recording,
            selection=selection,
            random_seed=random_seed,
        )
        print(f"[animal_size] wrote {walk_instances_path}")
        print(f"[animal_size] wrote {rearing_instances_path}")
