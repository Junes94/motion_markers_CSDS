from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd


def list_csvs(folder: Path | str) -> List[Path]:
    p = Path(folder)
    return sorted([q for q in p.glob("*.csv") if q.is_file()])


def _apply_labels(df: pd.DataFrame, labels: Sequence[str] | None) -> pd.DataFrame:
    if labels is None:
        return df
    if len(labels) > df.shape[1]:
        # More labels than columns; keep original columns
        return df
    out = df.iloc[:, : len(labels)].copy()
    out.columns = list(labels)
    return out


def _exclude_keypoints(df: pd.DataFrame, exclude_keypoints: Sequence[str] | None, coord_suffixes: Sequence[str]) -> pd.DataFrame:
    if not exclude_keypoints:
        return df
    to_drop = []
    base_names = set(exclude_keypoints)
    for base in base_names:
        for suf in coord_suffixes:
            col = f"{base}{suf}"
            if col in df.columns:
                to_drop.append(col)
    if to_drop:
        df = df.drop(columns=[c for c in to_drop if c in df.columns])
    return df


def read_pose_csv(
    path: Path | str,
    *,
    labels: Sequence[str] | None = None,
    exclude_keypoints: Sequence[str] | None = None,
    coord_suffixes: Sequence[str] = ("_x", "_y", "_z"),
    has_header: bool | None = None,
) -> pd.DataFrame:
    # Decide header handling: if labels are provided, default to no header
    if has_header is None:
        header_arg = 0  # infer/default
        if labels is not None:
            header_arg = None
    else:
        header_arg = 0 if has_header else None

    df = pd.read_csv(path, header=header_arg)
    df = _apply_labels(df, labels)
    df = _exclude_keypoints(df, exclude_keypoints, coord_suffixes)
    return df


def load_pose_folder(
    folder: Path | str,
    *,
    labels: Sequence[str] | None = None,
    exclude_keypoints: Sequence[str] | None = None,
    coord_suffixes: Sequence[str] = ("_x", "_y", "_z"),
    has_header: bool | None = None,
) -> Tuple[List[pd.DataFrame], List[str]]:
    dataframes: List[pd.DataFrame] = []
    names: List[str] = []
    for csv in list_csvs(folder):
        dataframes.append(
            read_pose_csv(
                csv,
                labels=labels,
                exclude_keypoints=exclude_keypoints,
                coord_suffixes=coord_suffixes,
                has_header=has_header,
            )
        )
        names.append(os.path.splitext(csv.name)[0])
    return dataframes, names
