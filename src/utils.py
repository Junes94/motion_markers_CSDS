from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root_from(file: str) -> Path:
    return Path(file).resolve().parents[2]


def add_src_to_syspath() -> None:
    import sys

    here = Path(__file__).resolve()
    root = here.parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_yaml(path: Path | str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

