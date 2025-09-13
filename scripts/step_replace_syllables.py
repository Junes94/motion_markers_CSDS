from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def run(cfg: Dict[str, Any]) -> None:
    params = cfg.get("parameters", {})
    ab = params.get("ablation", {})

    root = Path(__file__).resolve().parents[1]

    input_csv_cfg = ab.get("input_csv")
    output_csv_cfg = ab.get("output_csv")
    exclude_syllables: List[int] = list(ab.get("exclude_syllables", []))
    random_seed = int(ab.get("random_seed", 42))

    if not input_csv_cfg or not output_csv_cfg:
        print("[replace_syllables] Skipping: set parameters.ablation.input_csv and output_csv in config.")
        return

    input_csv = Path(input_csv_cfg)
    if not input_csv.is_absolute():
        input_csv = (root / input_csv).resolve()
    out_cfg_path = Path(output_csv_cfg)
    if not out_cfg_path.is_absolute():
        out_cfg_path = (root / out_cfg_path).resolve()

    # Build filename that reflects excluded syllables: *_replace_syll_[a,b].csv
    excl_str = ",".join(str(x) for x in sorted(exclude_syllables))
    def _normalized_stem(stem: str) -> str:
        if stem.endswith("_replace_syll") or stem.endswith("_replace_syll_"):
            return stem[: stem.rfind("_replace_syll")]
        return stem

    if out_cfg_path.suffix.lower() == ".csv":
        parent = out_cfg_path.parent
        base_stem = _normalized_stem(out_cfg_path.stem)
        output_csv = parent / f"{base_stem}_replace_syll_[{excl_str}].csv"
    else:
        # Treat as directory
        parent = out_cfg_path
        base_stem = _normalized_stem(input_csv.stem)
        output_csv = parent / f"{base_stem}_replace_syll_[{excl_str}].csv"

    print(f"[replace_syllables] input={input_csv}")
    print(f"[replace_syllables] output={output_csv}")
    print(f"[replace_syllables] exclude_syllables={exclude_syllables}, seed={random_seed}")

    np.random.seed(random_seed)

    if not input_csv.exists():
        print(f"[replace_syllables] Input CSV not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    if "syllable" not in df.columns:
        print(f"[replace_syllables] Input CSV has no 'syllable' column: {input_csv}")
        return

    out_rows = []
    for name, sub in df.groupby("name"):
        excluded = sub[sub["syllable"].isin(exclude_syllables)]
        kept = sub[~sub["syllable"].isin(exclude_syllables)]
        print(f"{name}: replacing {len(excluded)} excluded rows with resampled kept rows")
        if kept.empty or len(excluded) == 0:
            out_rows.append(sub)
            continue
        replacement = kept.sample(n=len(excluded), replace=True, random_state=random_seed)
        out_rows.append(pd.concat([kept, replacement], ignore_index=True))

    out_df = pd.concat(out_rows, ignore_index=True) if out_rows else df
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} (rows: {len(out_df)})")
