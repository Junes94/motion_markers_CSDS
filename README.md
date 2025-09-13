# Kinematic Analysis Pipeline

This repository provides a clean, reproducible analysis pipeline for computing kinematic scalar features from pose trajectories and aggregating them into histogram data used for figures and downstream analysis.

## Repository Structure

```
Repo_for_journal/
├─ README.md                      # Overview and usage
├─ requirements.txt               # Python dependencies
├─ .gitignore                     # Ignore large/generated files
├─ configs/
│  └─ config.yaml                 # Project configuration (paths, params)
├─ src/
│  └─ paper_analysis/             # Reusable analysis code (package)
│     ├─ __init__.py
│     └─ utils.py
├─ scripts/
│  ├─ run_pipeline.py             # Orchestrates pipeline steps
│  ├─ step_preprocess.py          # Data cleaning / preprocessing
│  ├─ step_analyze.py             # Analysis / stats
│  └─ step_plot.py                # Figures / plots
├─ data/
│  ├─ raw/                        # Unmodified inputs (not committed)
│  │  └─ .gitkeep
│  ├─ interim/                    # Intermediate files (not committed)
│  │  └─ .gitkeep
│  └─ processed/                  # Cleaned data (not committed)
│     └─ .gitkeep
├─ results/
│  ├─ figures/                    # Figures/plots (not committed)
│  │  └─ .gitkeep
│  └─ tables/                     # Tables/CSVs (not committed)
│     └─ .gitkeep
└─ notebooks/                     # Optional Jupyter notebooks
   └─ .gitkeep
```

## Quickstart

1) Create and activate a virtual environment (recommended):

- Windows (PowerShell):
  - `python -m venv .venv`
  - `.venv\\Scripts\\Activate.ps1`
- macOS/Linux:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`

2) Install dependencies:

```
pip install -r requirements.txt
```

3) Place your pose CSVs into `data/pose_traj/` and a group index CSV with columns `name,group` into `data/SIT/SIratio.csv`. The pipeline reads from `data/` and writes outputs under `results/`.

4) Run the pipeline:

- Windows (PowerShell):
  - `python scripts/run_pipeline.py` (no extra setup needed)
- macOS/Linux:
  - `python scripts/run_pipeline.py`

You can run specific steps:

```
python scripts/run_pipeline.py --steps compute_scalars build_histograms
```

And point to a different config:

```
python scripts/run_pipeline.py --config configs/config.yaml
```

### Common CLI Overrides

You can override key paths at runtime (without editing the YAML):

```
python scripts/run_pipeline.py \
  --pose-dir data/pose_traj \
  --index-csv data/SIT/SIratio.csv \
  --results-dir results \
  --histogram-dir results/scalar_histograms \
  --figures-dir results/figures
```

### Quick Ablation Runs

- Use the ablation output as the scalars source and write histograms to a separate folder:

```
python scripts/run_pipeline.py --steps replace_syllables build_histograms --use-ablation
```

- By default, the histogram folder is suffixed with the replaced syllables, e.g., `results/scalar_histograms_replace_syll_[3,12]`.
- You can customize the suffix via `--ablation-tag TAG` if desired.

## Configuration

Edit `configs/config.yaml` to set paths and parameters:

- `paths.pose_dir`: folder with input pose CSVs (e.g., `data/pose_traj`).
- `paths.group_index_csv`: CSV with columns `name,group` to label mice.
- `paths.results_dir`: base results output folder (defaults to `results/`).
- `parameters.fps`: frames per second of recordings
- `parameters.smoothing_window`: window size for optional smoothing
- `parameters.variables`: which summary variables to produce and plot
- `parameters.bin_method`: histogram bin rule (`freedman_diaconis`, `sturges`, `scott`, or `manual`)
- `parameters.variable_bins`: optional per-variable bins overriding the rule. Value can be an integer (bin count across 1st–99th percentile) or an explicit list of edges.
- `analysis.steps`: default step order (e.g., `compute_scalars`, `build_histograms`)

Paths in the config may be relative (recommended) or absolute. Relative paths resolve to the repository root, so the project remains portable on other machines.

## Data Policy

- Large or private data should not be committed. This repo ignores `results/*` by default and expects input data under `data/`.

## Reproducibility Notes

- Dependencies are pinned in `requirements.txt`. For stricter reproducibility, consider a lockfile or Conda `environment.yml`.

## Adapting to Your Project

- Configure labels and keypoint selection in `configs/config.yaml`:
  - `parameters.labels`: optional list of column labels to apply to input CSVs (useful when files are unlabeled).
  - `parameters.labels_file`: alternative path to a text file with one label per line.
  - `parameters.pose_has_header`: set to `true` if your CSVs include a header row; set to `false` for raw numeric files without headers.
  - `parameters.exclude_keypoints`: list of base keypoint names to drop (e.g., `tail`, `RF`, `LF`). Columns matching `<name>_x`, `<name>_y`, `<name>_z` are removed before analysis.
  - `parameters.coord_suffixes`: coordinate suffixes to use for labeling/filtering (default: `_x,_y,_z`). Set to 2D if needed.

- Required columns depend on which variables you compute. Defaults require `head`, `torso`, and `anus` keypoints for length/angle features.

### Ablation: Replace Syllables and Rebuild

- Configure ablation in `configs/config.yaml` under `parameters.ablation`.
- Run the replacement step:

```
python scripts/run_pipeline.py --steps replace_syllables
```

- Point histogram step to the ablation output by setting `paths.scalars_csv` to the ablation `output_csv` path, then run:

```
python scripts/run_pipeline.py --steps build_histograms
```

- Or in one go (override via CLI):

```
python scripts/run_pipeline.py --steps replace_syllables build_histograms \
  --results-dir results \
  --index-csv data/SIT/SIratio.csv
```
