"""Leakage-safe preprocessing.

Every transformer here learns its statistics in `fit()` only. They are always
used inside an sklearn Pipeline, so during cross-validation they are re-fitted
on each training fold and never see the validation fold (or the test set).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ConstantColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns that are constant in the data seen during fit (cd_000)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns)
        keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
        self.kept_columns_ = keep
        self.dropped_columns_ = [c for c in X.columns if c not in set(keep)]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.kept_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_columns_)


def build_tree_preprocessor():
    """Preprocessing for models that handle NaN natively (gradient boosting).

    Missing values are left as NaN so the model can learn from missingness.
    """

    return Pipeline([("drop_constant", ConstantColumnDropper())])


def build_imputing_preprocessor(scale=False):
    """Median imputation + missing-value indicator columns.

    Missingness is strongly associated with failure, so `add_indicator=True`
    keeps that signal instead of silently erasing it. Medians are learned from
    the training fold only.
    """

    steps = [
        ("drop_constant", ConstantColumnDropper()),
        (
            "impute",
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=False),
        ),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))

    return Pipeline(steps)
