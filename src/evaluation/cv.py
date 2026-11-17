"""Cross-validated out-of-fold (OOF) evaluation.

Leakage controls:
* The whole pipeline (preprocessing + model) is cloned and re-fitted inside
  every fold, so imputation medians, dropped columns, etc. come from the
  training fold only.
* Folds are stratified (only ~1.7% of rows are positive).
* APS rows are independent truck snapshots with no vehicle id, so plain
  stratified K-fold is appropriate here. (A vehicle-grouped split would be
  required for datasets with several rows per vehicle.)
* The decision threshold is chosen on OOF scores of the training data, never
  on the official test set.
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import aps_cost, best_cost_threshold, classification_summary


def oof_predict_proba(pipeline, X, y, n_splits=5, seed=42):
    """Return (oof_scores, fold_ids). Each row is scored by a model that never
    saw it during fitting."""

    y = np.asarray(y).astype(int)
    oof = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        model = clone(pipeline)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        fold_ids[val_idx] = fold

    if np.isnan(oof).any():
        raise RuntimeError("Some rows did not receive an out-of-fold prediction.")

    return oof, fold_ids


def cross_fitted_cost(y, oof, fold_ids):
    """APS cost where each fold's threshold is chosen on the OTHER folds only.

    This is the honest estimate: tuning a threshold and scoring it on the same
    rows is optimistic.
    """

    y = np.asarray(y).astype(int)
    total = 0
    thresholds = []

    for fold in np.unique(fold_ids):
        in_fold = fold_ids == fold
        threshold, _ = best_cost_threshold(y[~in_fold], oof[~in_fold])
        thresholds.append(threshold)
        total += aps_cost(y[in_fold], (oof[in_fold] >= threshold).astype(int))

    return int(total), [float(t) for t in thresholds]


def evaluate_pipeline_cv(pipeline, X, y, n_splits=5, seed=42):
    """Full CV evaluation of one pipeline. Training data only."""

    oof, fold_ids = oof_predict_proba(pipeline, X, y, n_splits=n_splits, seed=seed)

    tuned_threshold, tuned_cost = best_cost_threshold(y, oof)
    honest_cost, fold_thresholds = cross_fitted_cost(y, oof, fold_ids)

    return {
        "at_0.5": classification_summary(y, oof, threshold=0.5),
        "at_cost_optimal_threshold": classification_summary(y, oof, threshold=tuned_threshold),
        "cost_optimal_threshold": tuned_threshold,
        "cost_tuned_on_same_oof_OPTIMISTIC": tuned_cost,
        "cost_cross_fitted_threshold": honest_cost,
        "fold_thresholds": fold_thresholds,
        "n_splits": n_splits,
        "seed": seed,
    }
