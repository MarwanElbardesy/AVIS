import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.data import scania
from src.evaluation.cv import cross_fitted_cost, evaluate_pipeline_cv, oof_predict_proba
from src.models.baseline import make_baseline_models
from src.models import train_baselines
from tests.conftest import make_aps_like, write_aps_csv


@pytest.fixture(scope="module")
def models():
    return make_baseline_models(seed=0, fast=True)


def test_oof_covers_every_row_exactly_once(xy, models):
    X, y = xy
    oof, folds = oof_predict_proba(models["hist_gradient_boosting"], X, y, n_splits=4, seed=0)
    assert not np.isnan(oof).any()
    assert sorted(np.unique(folds)) == [0, 1, 2, 3]
    assert ((oof >= 0) & (oof <= 1)).all()


def test_folds_are_stratified(xy, models):
    X, y = xy
    _, folds = oof_predict_proba(models["logistic_regression"], X, y, n_splits=4, seed=0)
    positives_per_fold = [int(y.values[folds == f].sum()) for f in range(4)]
    assert max(positives_per_fold) - min(positives_per_fold) <= 1


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "hist_gradient_boosting"])
def test_models_learn_real_signal(xy, models, name):
    X, y = xy
    oof, _ = oof_predict_proba(models[name], X, y, n_splits=4, seed=0)
    assert roc_auc_score(y, oof) > 0.8


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "hist_gradient_boosting"])
def test_shuffled_labels_give_chance_level_auc(xy, models, name):
    """Leakage canary: with labels shuffled there is nothing to learn, so any
    AUC well above 0.5 would mean information is leaking across folds."""
    X, y = xy
    y_shuffled = np.random.default_rng(1).permutation(y.values)
    oof, _ = oof_predict_proba(models[name], X, y_shuffled, n_splits=4, seed=0)
    assert roc_auc_score(y_shuffled, oof) < 0.65


def test_cross_fitted_cost_not_better_than_optimistic(xy, models):
    X, y = xy
    res = evaluate_pipeline_cv(models["hist_gradient_boosting"], X, y, n_splits=4, seed=0)
    assert res["cost_cross_fitted_threshold"] >= res["cost_tuned_on_same_oof_OPTIMISTIC"]


def test_cross_fitted_cost_uses_other_folds_only():
    y = np.array([0, 0, 0, 0, 1, 0])
    oof = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.5])
    folds = np.array([0, 0, 0, 1, 1, 1])
    total, thresholds = cross_fitted_cost(y, oof, folds)
    # fold 0: threshold learned from fold 1 alone is 0.9 -> no alarms, cost 0.
    assert thresholds[0] == pytest.approx(0.9)
    # fold 1: fold 0 has no positives, so its "predict nobody" threshold sits
    # just above fold 0's max score (0.3). Applied to fold 1 it flags 0.4/0.9/0.5:
    # 2 false alarms (20) + the true positive caught = 20. Proves the threshold
    # came from the other fold, not from fold 1's own labels (which would give 0).
    assert thresholds[1] == pytest.approx(0.3, abs=1e-6)
    assert total == 20


def test_pipeline_results_are_reproducible(xy, models):
    X, y = xy
    a, _ = oof_predict_proba(models["hist_gradient_boosting"], X, y, n_splits=3, seed=7)
    b, _ = oof_predict_proba(models["hist_gradient_boosting"], X, y, n_splits=3, seed=7)
    assert np.allclose(a, b)


def test_train_script_runs_without_test_file(tmp_path, monkeypatch, capsys):
    train_path = write_aps_csv(make_aps_like(n=400, seed=3), tmp_path / "train.csv")
    monkeypatch.setattr(scania, "TEST_FILE", tmp_path / "must_not_be_read.csv")
    out = tmp_path / "out" / "cv.json"

    results = train_baselines.run(
        train_path=train_path, n_splits=3, seed=0, fast=True, only=["hist_gradient_boosting"], out_path=out
    )

    assert out.exists()
    saved = json.loads(out.read_text())
    assert "hist_gradient_boosting" in saved["models"]
    assert saved["data_audit"]["constant_columns"] == ["cd_000"]
    assert results["models"]["hist_gradient_boosting"]["n_splits"] == 3
