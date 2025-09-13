from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List


def _add_src_to_path() -> None:
    # Make `src/` importable without external env setup
    this = Path(__file__).resolve()
    root = this.parents[1]
    src = root / "src"
    # Ensure repo root is on sys.path so `scripts.*` imports work
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

from paper_analysis.utils import ensure_dir, load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run analysis pipeline")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "config.yaml"),
        help="Path to YAML config file",
    )
    # Optional CLI overrides for common paths
    p.add_argument("--pose-dir", type=str, default=None, help="Override paths.pose_dir")
    p.add_argument("--index-csv", type=str, default=None, help="Override paths.group_index_csv")
    p.add_argument("--results-dir", type=str, default=None, help="Override paths.results_dir")
    p.add_argument("--histogram-dir", type=str, default=None, help="Override paths.histogram_dir")
    p.add_argument("--figures-dir", type=str, default=None, help="Override paths.figures_dir")
    p.add_argument(
        "--steps",
        nargs="*",
        default=None,
        help="Subset of steps to run (default: config.analysis.steps)",
    )
    # Convenience ablation switches
    p.add_argument("--use-ablation", action="store_true", help="Use parameters.ablation.output_csv as scalars source and write histograms to a separate folder")
    p.add_argument("--ablation-tag", type=str, default="ablation", help="Suffix/tag for histogram output folder when --use-ablation is set")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    # Apply CLI overrides into cfg before resolving paths
    paths = cfg.get("paths", {})
    if args.pose_dir:
        paths["pose_dir"] = args.pose_dir
    if args.index_csv:
        paths["group_index_csv"] = args.index_csv
    if args.results_dir:
        paths["results_dir"] = args.results_dir
    if args.histogram_dir:
        paths["histogram_dir"] = args.histogram_dir
    if args.figures_dir:
        paths["figures_dir"] = args.figures_dir
    cfg["paths"] = paths

    default_steps: List[str] = cfg.get("analysis", {}).get("steps", ["preprocess", "analyze", "plot"])  # type: ignore
    steps: List[str] = args.steps if args.steps else default_steps

    print(f"Using config: {args.config}")
    print(f"Steps: {steps}")

    # If requested, configure ablation-derived inputs/outputs
    if args.use_ablation:
        ab = cfg.get("parameters", {}).get("ablation", {})
        out_csv = ab.get("output_csv")
        if out_csv:
            # Point scalars_csv to ablation output
            cfg.setdefault("paths", {})["scalars_csv"] = out_csv
        # Choose a meaningful suffix for histogram_dir
        excl = ab.get("exclude_syllables", []) or []
        if args.ablation_tag and args.ablation_tag != "ablation":
            suffix = args.ablation_tag
        else:
            # Default: reflect replaced syllables like the ablation CSV
            excl_str = ",".join(str(x) for x in sorted(excl))
            suffix = f"replace_syll_[{excl_str}]" if excl_str else "ablation"
        base_hist = cfg.get("paths", {}).get("histogram_dir", "results/scalar_histograms")
        from pathlib import Path as _P
        base_hist_p = _P(base_hist)
        tagged = base_hist_p.with_name(base_hist_p.name + f"_{suffix}")
        cfg["paths"]["histogram_dir"] = str(tagged)

    # Ensure folder structure exists (resolve relative to project root)
    root = Path(__file__).resolve().parents[1]
    paths = cfg.get("paths", {})
    for key in ["results_dir", "figures_dir", "tables_dir", "histogram_dir", "pose_dir", "group_index_csv", "scalars_csv"]:
        if key in paths and paths[key]:
            p = Path(paths[key])
            if not p.is_absolute():
                p = (root / p).resolve()
            ensure_dir(p if p.suffix == "" else p.parent)
            paths[key] = str(p)
    cfg["paths"] = paths

    # Lazy import step modules
    from importlib import import_module

    for step in steps:
        mod = None
        errors = []
        for mod_name in (f"scripts.step_{step}", f"step_{step}"):
            try:
                mod = import_module(mod_name)
                break
            except ModuleNotFoundError as e:
                errors.append(str(e))
        if mod is None:
            print(f"[WARN] Could not import step '{step}'. Tried: scripts.step_{step}, step_{step}. Errors: {errors}")
            continue

        if not hasattr(mod, "run"):
            print(f"[WARN] Step module '{mod_name}' missing a 'run(cfg)' function")
            continue

        print(f"\n==> Running step: {step}")
        mod.run(cfg)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
