# Opstimus

Opstimus is a thesis-oriented repository for anomaly detection and lightweight root cause analysis on tabular and multivariate machine data. The current codebase focuses on building a reusable pipeline that can start from one demonstration dataset, then scale toward additional datasets such as credit card fraud and SMD.

## Current Scope

- Modular anomaly detectors: Isolation Forest, LOF, Autoencoder
- Dataset adapters for credit card and SMD
- Config-driven pipeline entrypoint
- Detection metrics and artifact export
- A baseline RCA module based on feature contribution ranking

## Repository Layout

- `datasets/`: dataset-specific loaders and adapters
- `preprocessing/`: shared preprocessing utilities
- `detection/`: anomaly detection models
- `evaluation/`: metrics and report writers
- `rca/`: root cause ranking logic
- `pipelines/`: end-to-end orchestration
- `config/`: runnable experiment configurations
- `experiments/` and `notebooks/`: exploratory and legacy experiment code

## Installation

```bash
pip install -r requirements.txt
```

## Run The Pipeline

Example with the credit card baseline:

```bash
python main.py --config config/credit_card/isolation_forest.json
```

The pipeline will:

1. Load the dataset using a dataset adapter.
2. Scale train and test features.
3. Train the configured detector.
4. Generate anomaly scores and predictions.
5. Compute detection metrics when labels are available.
6. Rank likely root-cause features for detected anomalies.
7. Save outputs to `artifacts/...`.

## Run The Demo Dashboard

After generating artifacts, launch the local dashboard:

```bash
venv_opstimus\Scripts\python.exe visualization\dashboard.py --port 8765
```

Or on Windows:

```bash
run_dashboard_demo.bat
```

Then open:

```text
http://127.0.0.1:8765
```

The dashboard reads existing outputs in `artifacts/` and shows:

- detection metrics
- anomaly score trend
- prediction vs ground truth
- global RCA ranking
- segment-level RCA
- event-level RCA matches for SMD

## Run Multiple Configs

To execute all JSON configs under `config/` and build a leaderboard:

```bash
venv_opstimus\Scripts\python.exe run_batch.py --config-root config --output-dir artifacts/batch_runs
```

Outputs:

- `artifacts/batch_runs/leaderboard.csv`
- `artifacts/batch_runs/batch_summary.json`

The batch runner is fault-tolerant: one bad or missing dataset config will be marked as failed without stopping the whole batch.

## Output Artifacts

Each run writes the following files into the configured output directory:

- `summary.json`: dataset metadata, detector config, metrics, and RCA summary
- `predictions.csv`: anomaly scores, predictions, and labels if available
- `root_causes.csv`: ranked contributing features
- serialized model and scaler files

## Notes For Thesis Development

- The current RCA implementation is a baseline feature-ranking approach, not yet a full causal RCA method.
- Notebook experiments are still available, but the new recommended execution path is the config-driven pipeline.
- The next natural step is to add stronger RCA methods and time-series event-level evaluation.
