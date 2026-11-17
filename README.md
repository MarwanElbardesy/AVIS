# CVI - Commercial Vehicle Intelligence

Predictive-maintenance intelligence for heavy trucks, built on the Scania APS Failure dataset
(air pressure system failures). Goal: reliable failure prediction, explainability, maintenance
recommendations and an interactive dashboard.

## Status
Milestone 1 (leakage-safe ML foundation) is implemented; Milestone 2 (EDA) is in progress (2a, 2b, 2c done; 2d report in progress (2d-1 done)). See `PROGRESS.md` and `DECISIONS.md`.
No model results are published in this repo until they are produced by running the code on the real data.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Place the official files (not committed) in `data/raw/scania/`:
`aps_failure_training_set.csv`, `aps_failure_test_set.csv`.

## Run
```powershell
python -m pytest                       # unit tests (synthetic data only)
python -m src.models.train_baselines   # 5-fold CV on the TRAINING set only
python -m src.eda.run_eda              # EDA on the TRAINING set only -> reports/eda/eda_results.json
```

## Evaluation protocol
* Official metric: cost = 10 x false positives + 500 x false negatives.
* All development uses stratified 5-fold CV on the training set. Preprocessing is inside the
  per-fold pipeline, so no statistics leak from validation rows.
* Thresholds are chosen on out-of-fold scores; the headline cost uses a *cross-fitted* threshold
  (chosen without the evaluated fold).
* The official test set is loaded only by `load_scania_test()`, which no development code calls.
