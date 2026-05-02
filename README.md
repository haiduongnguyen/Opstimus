# Opstimus

Opstimus is a workflow-based anomaly detection and root cause analysis system for tabular and multivariate time-series datasets. The repo now supports two operating modes from a single JSON config:

- `train`: benchmark multiple models, run RCA, score the runs, select the best model, and save it for reuse
- `inference`: run anomaly detection + RCA on data-only input, either from preset models or from a saved best model

## Repository Layout

- `main.py`: single entrypoint for all workflows
- `config/`: workflow configs and templates
- `workflows/`: workflow config loader, profiles, executors, train/inference orchestration
- `datasets/`: dataset adapters
- `detection/`: anomaly detector implementations
- `thresholding/`: threshold strategies
- `rca/`: RCA logic and reference profile handling
- `evaluation/`: metrics and JSON/CSV report writing
- `visualization/`: local dashboard

## Install

```bash
pip install -r requirements.txt
```

## Workflow Model

### 1. Train Mode

Use when you have training data or a dataset preset that contains train/test split.

What it does:

1. Load dataset
2. Expand a `model_profile` into multiple candidate detectors
3. Run multiple threshold profiles per detector
4. Compute detection metrics and RCA
5. Build `leaderboard.csv`
6. Select the best run by a configured metric
7. Copy the best run into `best_model/`

Example:

```bash
venv_opstimus\Scripts\python.exe main.py --config config/train_smd_machine_1_1.json
```

### 2. Inference Mode

Use when you only have input data and want anomaly + RCA output.

Two ways:

- `model_source = "profile"`: run a preset list of unsupervised models on the input data
- `model_source = "saved_run"`: load a previously selected best model and reuse it

Example:

```bash
venv_opstimus\Scripts\python.exe main.py --config config/inference_smd_machine_1_1_profile.json
```

## Config Philosophy

Each workflow uses one JSON file. You declare:

- `mode`
- `dataset`
- `benchmark` or `inference`
- `selection`
- `rca`
- `deployment`

### Train Config Example

```json
{
  "mode": "train",
  "workflow": {
    "name": "train_smd_machine_1_1"
  },
  "dataset": "smd.machine_1_1",
  "benchmark": {
    "model_profile": "time_series_baseline",
    "threshold_profiles": ["default", "percentile_97"]
  },
  "selection": {
    "metric": "f1",
    "higher_is_better": true
  }
}
```

### Inference Config Example

```json
{
  "mode": "inference",
  "dataset": "csv_data_only",
  "dataset_overrides": {
    "data_path": "data/raw/your_data.csv",
    "label_col": null
  },
  "inference": {
    "model_source": "profile",
    "model_profile": "tabular_inference_fast",
    "threshold_profiles": ["default", "percentile_95"]
  }
}
```

## Presets and Profiles

Dataset presets are defined in [workflows/profiles.py](/D:/thac_si_phenika/master_thesis/Opstimus/workflows/profiles.py).

Current dataset presets:

- `smd.machine_1_1`
- `sklearn_breast_cancer`
- `credit_card`
- `csv_data_only`

Current model profiles:

- `tabular_baseline`
- `time_series_baseline`
- `tabular_inference_fast`
- `time_series_inference_fast`

Current threshold profiles:

- `default`
- `percentile_95`
- `percentile_97`
- `stddev_3`

## Example Configs

- [train_smd_machine_1_1.json](/D:/thac_si_phenika/master_thesis/Opstimus/config/train_smd_machine_1_1.json)
- [train_sklearn_breast_cancer.json](/D:/thac_si_phenika/master_thesis/Opstimus/config/train_sklearn_breast_cancer.json)
- [inference_smd_machine_1_1_profile.json](/D:/thac_si_phenika/master_thesis/Opstimus/config/inference_smd_machine_1_1_profile.json)
- [inference_sklearn_from_saved_best.json](/D:/thac_si_phenika/master_thesis/Opstimus/config/inference_sklearn_from_saved_best.json)
- [inference_csv_data_only_template.json](/D:/thac_si_phenika/master_thesis/Opstimus/config/inference_csv_data_only_template.json)
- [inference_from_saved_best_template.json](/D:/thac_si_phenika/master_thesis/Opstimus/config/inference_from_saved_best_template.json)

## Output Structure

Each workflow writes to:

```text
artifacts/workflows/<workflow_name>/
```

Typical contents:

- `workflow_summary.json`
- `leaderboard.csv`
- `runs/<run_id>/summary.json`
- `runs/<run_id>/predictions.csv`
- `runs/<run_id>/root_causes.csv`
- `runs/<run_id>/reference_profile.json`
- `best_model/` for train workflows

## Local Dashboard

Start:

```bash
venv_opstimus\Scripts\python.exe visualization\dashboard.py --port 8765
```

Or:

```bash
run_dashboard_demo.bat
```

The dashboard scans `artifacts/` and can open workflow runs directly, including new benchmark runs under `artifacts/workflows/...`.

## Notes

- RCA is still a contribution-based baseline, not causal RCA.
- `train` mode is the right path for a new dataset when you do not know which detector is best.
- `inference` mode is the right path for data-only analysis or for reusing a selected best model.
