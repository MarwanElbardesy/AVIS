"""Baseline model zoo for APS failure prediction.

Three deliberately different approaches so the comparison is meaningful:
* logistic regression  - linear, needs imputation + scaling
* random forest        - bagged trees, needs imputation (+ missing indicators)
* hist gradient boosting - boosted trees, handles NaN natively
"""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocessing import (
    build_imputing_preprocessor,
    build_tree_preprocessor,
)


def make_baseline_models(seed=42, fast=False):
    """Return {name: unfitted Pipeline}. `fast=True` shrinks models for tests."""

    n_trees = 20 if fast else 300
    boost_iter = 30 if fast else 300

    return {
        "logistic_regression": Pipeline(
            [
                ("prep", build_imputing_preprocessor(scale=True)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, C=0.1, random_state=seed
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", build_imputing_preprocessor(scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=n_trees,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", build_tree_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=boost_iter,
                        learning_rate=0.05,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
