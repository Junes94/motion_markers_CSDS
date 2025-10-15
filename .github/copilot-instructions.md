# Copilot Instructions for motion_markers_CSDS

## Project Overview
- This repository implements a reproducible pipeline for kinematic and motion syllable analysis from pose trajectory CSVs.
- The main workflow is orchestrated by `scripts/run_pipeline.py`, which chains together preprocessing, scalar computation, histogram building, and ablation steps.
- Configuration is centralized in `configs/config.yaml` (paths, parameters, variables, binning, ablation, etc.).
- Data flows: `data/pose_traj/` (input CSVs) → pipeline steps → `results/` (outputs: histograms, figures, tables).

## Key Components
- `scripts/`: Pipeline entrypoints. Each `step_*.py` script implements a modular stage (preprocess, compute scalars, build histograms, replace syllables).
- `src/`: Core logic (feature extraction, I/O, utilities). Example: `src/features.py` for kinematic feature computation.
- `configs/`: Main config YAML and label files. All paths are relative to repo root.
- `data/`: Input data (not committed). `pose_traj/` for pose CSVs, `SIT/` for group index.

## Developer Workflows
- **Setup:**
  - Use a virtual environment. Install dependencies with `pip install -r requirements.txt`.
- **Run pipeline:**
  - `python scripts/run_pipeline.py` (runs all steps as per config)
  - To run specific steps: `python scripts/run_pipeline.py --steps compute_scalars build_histograms`
  - Override config at runtime: `python scripts/run_pipeline.py --config configs/config.yaml`
  - Override paths: `python scripts/run_pipeline.py --pose-dir ... --index-csv ... --results-dir ...`
- **Ablation:**
  - `python scripts/run_pipeline.py --steps replace_syllables build_histograms --use-ablation`
  - Customize ablation via config or CLI (`--ablation-tag`)

## Project Conventions
- All paths in config are relative to repo root for portability.
- Input CSVs must match expected keypoint/label structure (see config for details).
- Large/intermediate data is ignored by git; only code and configs are versioned.
- Modular step scripts: add new steps by following the `step_*.py` pattern.

## Integration & Extensibility
- Add new features in `src/features.py` and expose via pipeline steps.
- Update `configs/config.yaml` to add new variables, bins, or analysis steps.
- For new data types, update I/O logic in `src/io.py` and adjust config accordingly.

## Examples
- Run full pipeline: `python scripts/run_pipeline.py`
- Run only histogram step: `python scripts/run_pipeline.py --steps build_histograms`
- Use custom config: `python scripts/run_pipeline.py --config configs/config.yaml`

Refer to `README.md` and `configs/config.yaml` for further details and up-to-date usage patterns.
