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
python main.py --config config/credit_card_isolation_forest.json
```

The pipeline will:

1. Load the dataset using a dataset adapter.
2. Scale train and test features.
3. Train the configured detector.
4. Generate anomaly scores and predictions.
5. Compute detection metrics when labels are available.
6. Rank likely root-cause features for detected anomalies.
7. Save outputs to `artifacts/...`.

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
