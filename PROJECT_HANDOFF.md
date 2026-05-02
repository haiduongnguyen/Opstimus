# Opstimus Project Handoff

## Current Direction

The project is now workflow-centric, not run-centric.

Main goal:

- one JSON config per workflow
- support many datasets
- support both `train` and `inference` modes
- combine anomaly detection with RCA
- save reusable best models
- expose results through local dashboard

## Operating Modes

### `train`

Use when train data exists.

Behavior:

1. load dataset
2. expand `model_profile` into multiple candidate models
3. run threshold variants
4. compute detection metrics + RCA
5. build leaderboard
6. select best run
7. copy best run into `best_model/`

Example config:

- [train_smd_machine_1_1.json](/D:/thac_si_phenika/master_thesis/Opstimus/config/train_smd_machine_1_1.json)
- [train_sklearn_breast_cancer.json](/D:/thac_si_phenika/master_thesis/Opstimus/config/train_sklearn_breast_cancer.json)

### `inference`

Use when only input data is available.

Two branches:

- `model_source = "profile"`: run preset unsupervised models directly on input data
- `model_source = "saved_run"`: load a trained best model and reuse it

Example config:

- [inference_smd_machine_1_1_profile.json](/D:/thac_si_phenika/master_thesis/Opstimus/config/inference_smd_machine_1_1_profile.json)
- [inference_sklearn_from_saved_best.json](/D:/thac_si_phenika/master_thesis/Opstimus/config/inference_sklearn_from_saved_best.json)
- [inference_csv_data_only_template.json](/D:/thac_si_phenika/master_thesis/Opstimus/config/inference_csv_data_only_template.json)

## Main Entry Point

- [main.py](/D:/thac_si_phenika/master_thesis/Opstimus/main.py)

Run:

```bash
venv_opstimus\Scripts\python.exe main.py --config config/train_smd_machine_1_1.json
```

## Key Packages

### Workflow Layer

- [workflows/loader.py](/D:/thac_si_phenika/master_thesis/Opstimus/workflows/loader.py)
  - loads and normalizes workflow config
- [workflows/profiles.py](/D:/thac_si_phenika/master_thesis/Opstimus/workflows/profiles.py)
  - dataset presets, model profiles, threshold profiles
- [workflows/executor.py](/D:/thac_si_phenika/master_thesis/Opstimus/workflows/executor.py)
  - single-run execution, artifact save/load, RCA reference profile handling
- [workflows/train.py](/D:/thac_si_phenika/master_thesis/Opstimus/workflows/train.py)
  - benchmark, leaderboard, best-model selection
- [workflows/inference.py](/D:/thac_si_phenika/master_thesis/Opstimus/workflows/inference.py)
  - profile-based inference or saved-model inference
- [workflows/runner.py](/D:/thac_si_phenika/master_thesis/Opstimus/workflows/runner.py)
  - dispatch by mode

### Dataset Layer

- [datasets/base.py](/D:/thac_si_phenika/master_thesis/Opstimus/datasets/base.py)
- [datasets/registry.py](/D:/thac_si_phenika/master_thesis/Opstimus/datasets/registry.py)
- [datasets/csv_data_only.py](/D:/thac_si_phenika/master_thesis/Opstimus/datasets/csv_data_only.py)
- [datasets/smd.py](/D:/thac_si_phenika/master_thesis/Opstimus/datasets/smd.py)
- [datasets/sklearn_breast_cancer.py](/D:/thac_si_phenika/master_thesis/Opstimus/datasets/sklearn_breast_cancer.py)
- [datasets/credit_card.py](/D:/thac_si_phenika/master_thesis/Opstimus/datasets/credit_card.py)

### Model / RCA / Evaluation

- [detection/isolation_forest.py](/D:/thac_si_phenika/master_thesis/Opstimus/detection/isolation_forest.py)
- [detection/lof.py](/D:/thac_si_phenika/master_thesis/Opstimus/detection/lof.py)
- [detection/autoencoder.py](/D:/thac_si_phenika/master_thesis/Opstimus/detection/autoencoder.py)
- [thresholding/strategies.py](/D:/thac_si_phenika/master_thesis/Opstimus/thresholding/strategies.py)
- [rca/feature_ranking.py](/D:/thac_si_phenika/master_thesis/Opstimus/rca/feature_ranking.py)
- [evaluation/metrics.py](/D:/thac_si_phenika/master_thesis/Opstimus/evaluation/metrics.py)

## Current Presets

### Dataset Presets

- `smd.machine_1_1`
- `sklearn_breast_cancer`
- `credit_card`
- `csv_data_only`

Defined in:

- [workflows/profiles.py](/D:/thac_si_phenika/master_thesis/Opstimus/workflows/profiles.py)

### Model Profiles

- `tabular_baseline`
- `time_series_baseline`
- `tabular_inference_fast`
- `time_series_inference_fast`

### Threshold Profiles

- `default`
- `percentile_95`
- `percentile_97`
- `stddev_3`

## Artifact Layout

Each workflow writes to:

```text
artifacts/workflows/<workflow_name>/
```

Main files:

- `workflow_summary.json`
- `leaderboard.csv`
- `runs/<run_id>/summary.json`
- `runs/<run_id>/predictions.csv`
- `runs/<run_id>/root_causes.csv`
- `runs/<run_id>/reference_profile.json`
- `best_model/` for train workflows

## Dashboard

- [visualization/dashboard.py](/D:/thac_si_phenika/master_thesis/Opstimus/visualization/dashboard.py)

Start:

```bash
run_dashboard_demo.bat
```

or

```bash
venv_opstimus\Scripts\python.exe visualization\dashboard.py --port 8765
```

Dashboard now reads workflow run directories under `artifacts/workflows/...` and can open runs from leaderboard rows using `run_dir`.

## What Was Removed

The old run-centric layer was removed to keep the repo clean:

- old `pipelines/`
- old `run_batch.py`
- old per-detector JSON configs under nested dataset folders
- old Python `config/*.py` package

## Smoke Tests Already Run

Successful:

- `main.py --config config/train_sklearn_breast_cancer.json`
- `main.py --config config/inference_smd_machine_1_1_profile.json`
- `main.py --config config/inference_sklearn_from_saved_best.json`

Observed behavior:

- train workflow created leaderboard, selected best run, and wrote `best_model/`
- inference workflow ran in data-only mode using profile models
- saved-model inference loaded `best_model/` and produced anomaly + RCA outputs

## Important Notes

- RCA is still contribution-based, not causal.
- `credit_card` preset still depends on local raw file `data/raw/creditcard/creditcard.csv`.
- For a new dataset, the right path is:
  1. add or map dataset
  2. run `train` workflow with a profile
  3. inspect leaderboard
  4. reuse `best_model/` in `inference`

## Next Logical Improvements

1. Add event-level detection metrics for time series.
2. Add more dataset presets.
3. Add more generic CSV / time-series loaders.
4. Add richer selection policies, for example weighted detection + RCA score.
5. Add workflow-level dashboard page, not only run-level page.
