from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import os
import re
import csv

import pandas as pd


def sanitize_filename(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "variable"


def sort_cohorts(values: Sequence[object]) -> List[object]:
    def _key(x: object):
        sx = str(x)
        try:
            return (0, float(sx), sx)
        except ValueError:
            return (1, sx, sx)

    return sorted(list(values), key=_key)


def derive_cohort_from_id(id_value: object, pattern: re.Pattern, divisor: int) -> Optional[int]:
    if pd.isna(id_value):
        return None
    match = pattern.search(str(id_value))
    if not match:
        return None
    token = match.group(1)
    try:
        n = int(token)
    except ValueError:
        return None
    if divisor <= 0:
        return n
    return n // divisor


def write_distance_style_csv(
    out_path: str,
    cohorts: Sequence[object],
    group_rows: Sequence[str],
    data_map: dict,
    max_individuals: int,
) -> None:
    header = [""]
    for cohort in cohorts:
        header.extend([cohort] * max_individuals)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for group in group_rows:
            row = [group]
            for cohort in cohorts:
                values = data_map.get((group, cohort), [])
                values = values[:max_individuals]
                padded = values + [""] * (max_individuals - len(values))
                row.extend(padded)
            writer.writerow(row)


def run(cfg: Dict[str, Any]) -> None:
    paths = cfg.get("paths", {})
    params = cfg.get("parameters", {})
    root = Path(__file__).resolve().parents[1]

    histogram_dir = Path(paths.get("histogram_dir", "results/scalar_histograms"))
    if not histogram_dir.is_absolute():
        histogram_dir = (root / histogram_dir).resolve()

    src_ma = histogram_dir / "group_mouse_averages_all.csv"
    if not src_ma.exists():
        print(f"[export_cohort_csvs] source not found: {src_ma}")
        return

    df = pd.read_csv(src_ma)
    # expect columns: variable, group, name, mean, std, n
    if not {"variable", "group", "name", "mean"}.issubset(set(df.columns)):
        print(f"[export_cohort_csvs] unexpected columns in {src_ma}. Need variable, group, name, mean")
        return

    out_base = histogram_dir
    out_base.mkdir(parents=True, exist_ok=True)

    cohort_divisor = int(params.get("cohort_divisor", 100))
    cohort_regex = str(params.get("cohort_id_regex", r"(\d+)$"))
    pattern = re.compile(cohort_regex)

    all_groups = df["group"].dropna().astype(str).drop_duplicates().tolist()
    group_order_cfg = cfg.get("group_order", None)
    if not group_order_cfg:
        # try parameters or default
        group_order = all_groups
    else:
        # keep case mapping from original
        requested = [str(x) for x in group_order_cfg]
        requested_lower = [x.lower() for x in requested]
        mapping = {str(v).lower(): str(v) for v in all_groups}
        ordered: List[str] = []
        for g in requested_lower:
            if g in mapping:
                ordered.append(mapping[g])
        for v in [str(x) for x in all_groups]:
            if v not in ordered:
                ordered.append(v)
        group_order = ordered

    variables = df["variable"].drop_duplicates().sort_values().tolist()
    written = 0
    for var in variables:
        sub = df[df["variable"] == var].copy()
        if sub.empty:
            continue

        # derive cohorts
        sub["_cohort"] = sub["name"].apply(lambda x: derive_cohort_from_id(x, pattern, cohort_divisor))
        sub = sub.dropna(subset=["_cohort", "mean"]).copy()
        cohorts = sort_cohorts(sub["_cohort"].dropna().unique().tolist())
        if not cohorts:
            print(f"[SKIP] {var}: no cohort values")
            continue

        cohort_counts = sub.groupby("_cohort").size()
        summary = ", ".join(f"{cohort}={int(cohort_counts.get(cohort, 0))}" for cohort in cohorts)
        print(f"[INFO] {var}: cohort counts -> {summary}")

        # compute observed max individuals per (group, cohort)
        counts = sub.groupby(["group", "_cohort"]).size()
        observed_max = int(counts.max()) if not counts.empty else 0
        max_individuals = max(1, observed_max)

        data_map = {}
        for cohort in cohorts:
            for group in group_order:
                m = (sub["_cohort"] == cohort) & (sub["group"].str.lower() == str(group).lower())
                vals_df = sub.loc[m, ["name", "mean"]].copy()
                if vals_df.empty:
                    data_map[(group, cohort)] = []
                    continue
                vals_df = vals_df.sort_values(by="name")
                data_map[(group, cohort)] = vals_df["mean"].tolist()

        out_name = f"{sanitize_filename(var)}_cohort.csv"
        out_path = out_base / out_name
        write_distance_style_csv(
            out_path=str(out_path),
            cohorts=cohorts,
            group_rows=group_order,
            data_map=data_map,
            max_individuals=max_individuals,
        )
        written += 1
        print(f"[DONE] {var} -> {out_path}")

    print(f"[SUMMARY] wrote {written} files to: {out_base}")
