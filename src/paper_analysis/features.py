from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd


def compute_scalar_summary(
    df: pd.DataFrame,
    *,
    fps: int,
    origin: Tuple[float, float] = (0.0, 0.0),
    smoothing_window: int | None = None,
    centerpoint: Sequence[str] = ("head", "torso"),
    length_criteria: Sequence[str] = ("head", "anus"),
    height_criteria: Sequence[str] = ("head", "anus"),
    velocity_criteria: Sequence[str] = ("head", "torso"),
) -> pd.DataFrame:
    """Compute kinematic scalar features from keypoint trajectories.

    Returns a DataFrame with columns:
    distance_from_origin, velocity_xy, velocity_z, length, height, torso_angle, angle_to_origin
    """

    work = df.copy()
    if smoothing_window is not None and smoothing_window > 1:
        work = work.rolling(window=int(smoothing_window), min_periods=1, center=True).mean()

    cx = (work[f"{centerpoint[0]}_x"] + work[f"{centerpoint[1]}_x"]) / 2
    cy = (work[f"{centerpoint[0]}_y"] + work[f"{centerpoint[1]}_y"]) / 2
    cz = (work[f"{centerpoint[0]}_z"] + work[f"{centerpoint[1]}_z"]) / 2

    dist_from_origin = np.sqrt((cx - origin[0]) ** 2 + (cy - origin[1]) ** 2)

    vx = (work[f"{velocity_criteria[0]}_x"] + work[f"{velocity_criteria[1]}_x"]) / 2
    vy = (work[f"{velocity_criteria[0]}_y"] + work[f"{velocity_criteria[1]}_y"]) / 2
    vxy = np.sqrt(vx.diff() ** 2 + vy.diff() ** 2) / (1 / fps)

    vz = abs(((work[f"{velocity_criteria[0]}_z"] + work[f"{velocity_criteria[1]}_z"]) / 2).diff() / (1 / fps))

    length = np.sqrt(
        (work[f"{length_criteria[0]}_x"] - work[f"{length_criteria[1]}_x"]) ** 2
        + (work[f"{length_criteria[0]}_y"] - work[f"{length_criteria[1]}_y"]) ** 2
    )

    height = work[f"{height_criteria[0]}_z"] - work[f"{height_criteria[1]}_z"]

    head_torso_x = work["head_x"] - work["torso_x"]
    head_torso_y = work["head_y"] - work["torso_y"]
    torso_anus_x = work["anus_x"] - work["torso_x"]
    torso_anus_y = work["anus_y"] - work["torso_y"]

    dot_product = head_torso_x * torso_anus_x + head_torso_y * torso_anus_y
    cross_product = head_torso_x * torso_anus_y - head_torso_y * torso_anus_x
    torso_angle = np.abs(np.arctan2(cross_product, dot_product))

    origin_vec_x = origin[0] - work["torso_x"]
    origin_vec_y = origin[1] - work["torso_y"]
    dot_origin = head_torso_x * origin_vec_x + head_torso_y * origin_vec_y
    cross_origin = head_torso_x * origin_vec_y - head_torso_y * origin_vec_x
    angle_to_origin = np.abs(np.arctan2(cross_origin, dot_origin))

    out = pd.DataFrame(
        {
            "distance_from_origin": dist_from_origin,
            "velocity_xy": vxy,
            "velocity_z": vz,
            "angle_to_origin": angle_to_origin,
            "length": length,
            "height": height,
            "torso_angle": torso_angle,
        }
    )
    return out


def freedman_diaconis_bins(arr: np.ndarray) -> int:
    data = arr[np.isfinite(arr)]
    if data.size < 2:
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return 10
    h = 2 * iqr * (data.size ** (-1 / 3))
    if h <= 0:
        return 10
    n_bins = int(np.ceil((data.max() - data.min()) / h))
    return max(n_bins, 1)


def sturges_bins(arr: np.ndarray) -> int:
    n = np.isfinite(arr).sum()
    return max(int(np.ceil(np.log2(max(n, 1))) + 1), 1)


def scott_bins(arr: np.ndarray) -> int:
    data = arr[np.isfinite(arr)]
    if data.size < 2:
        return 1
    sigma = np.std(data)
    if sigma == 0:
        return 10
    h = 3.5 * sigma * (data.size ** (-1 / 3))
    if h <= 0:
        return 10
    n_bins = int(np.ceil((data.max() - data.min()) / h))
    return max(n_bins, 1)
