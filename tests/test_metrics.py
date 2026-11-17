import numpy as np
import pytest

from src.evaluation.metrics import (
    FN_COST,
    FP_COST,
    aps_cost,
    best_cost_threshold,
    classification_summary,
)


def test_cost_weights():
    assert (FP_COST, FN_COST) == (10, 500)


def test_aps_cost_counts_fp_and_fn():
    y_true = [1, 1, 0, 0, 0]
    y_pred = [1, 0, 1, 1, 0]  # 1 FN, 2 FP
    assert aps_cost(y_true, y_pred) == 500 + 2 * 10


def test_perfect_predictions_cost_zero():
    y = [0, 1, 0, 1]
    assert aps_cost(y, y) == 0


def _brute_force(y, s):
    best = aps_cost(y, np.zeros_like(y))
    for t in np.unique(s):
        best = min(best, aps_cost(y, (s >= t).astype(int)))
    return best


@pytest.mark.parametrize("seed", range(5))
def test_best_threshold_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(300) < 0.1).astype(int)
    y[0] = 1
    s = np.round(rng.random(300) * 0.5 + y * 0.3, 2)  # rounding creates ties
    threshold, cost = best_cost_threshold(y, s)
    assert cost == _brute_force(y, s)
    assert aps_cost(y, (s >= threshold).astype(int)) == cost


def test_threshold_flags_everything_when_cheaper_than_missing_failure():
    y = np.array([1, 0, 0, 0])
    s = np.array([0.1, 0.9, 0.8, 0.7])  # positive ranked last
    threshold, cost = best_cost_threshold(y, s)
    assert cost == 30  # 3 false alarms (30) beats 1 missed failure (500)
    assert aps_cost(y, (s >= threshold).astype(int)) == 30


def test_threshold_predicts_nobody_when_false_alarms_cost_more():
    y = np.array([1] + [0] * 100)
    s = np.concatenate([[0.0], np.linspace(0.1, 1.0, 100)])  # positive ranked last
    threshold, cost = best_cost_threshold(y, s)
    assert cost == 500  # flagging all would cost 1000
    assert aps_cost(y, (s >= threshold).astype(int)) == 500


def test_threshold_input_validation():
    with pytest.raises(ValueError):
        best_cost_threshold([1, 0], [0.5])


def test_classification_summary_confusion_matrix():
    y = [1, 1, 0, 0]
    s = [0.9, 0.2, 0.8, 0.1]
    m = classification_summary(y, s, threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["aps_cost"] == 510
    assert 0 <= m["roc_auc"] <= 1
