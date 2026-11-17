"""Evaluation metrics, including the official Scania APS cost metric."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Official APS challenge costs: a false alarm costs a needless workshop
# inspection; a missed failure costs a breakdown on the road.
FP_COST = 10
FN_COST = 500


def aps_cost(y_true, y_pred, fp_cost=FP_COST, fn_cost=FN_COST):
    """Total APS cost = fp_cost * FP + fn_cost * FN (lower is better)."""

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return fp_cost * fp + fn_cost * fn


def best_cost_threshold(y_true, scores, fp_cost=FP_COST, fn_cost=FN_COST):
    """Find the score threshold (predict positive if score >= threshold) that
    minimises the APS cost. Returns (threshold, cost).

    Vectorised: sort by score and evaluate every distinct cut in one pass.
    Ties are never split, so the result matches a brute-force search.
    """

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape or y_true.size == 0:
        raise ValueError("y_true and scores must be non-empty and the same length.")

    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = y_true[order]

    tp_cum = np.cumsum(y)
    k = np.arange(1, len(y) + 1)  # number predicted positive
    fp = k - tp_cum
    fn = y.sum() - tp_cum
    cost = fp_cost * fp + fn_cost * fn

    # Only cut where the score changes (end of a tie group).
    last_of_group = np.append(s[1:] != s[:-1], True)
    cost = cost[last_of_group]
    thresholds = s[last_of_group]

    # "Predict nobody" baseline: all positives missed.
    no_positive_cost = fn_cost * int(y.sum())

    best = int(np.argmin(cost))
    if cost[best] <= no_positive_cost:
        return float(thresholds[best]), int(cost[best])

    return float(np.nextafter(s[0], np.inf)), int(no_positive_cost)


def classification_summary(y_true, scores, threshold=0.5):
    """Threshold-free and thresholded metrics in one dict."""

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    has_both = len(np.unique(y_true)) == 2

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, scores)) if has_both else None,
        "pr_auc": float(average_precision_score(y_true, scores)) if has_both else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "aps_cost": aps_cost(y_true, y_pred),
    }
