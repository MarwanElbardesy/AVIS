import numpy as np
import pandas as pd

from src.features.preprocessing import (
    ConstantColumnDropper,
    build_imputing_preprocessor,
    build_tree_preprocessor,
)


def test_constant_dropper_learns_from_fit_only():
    train = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    dropper = ConstantColumnDropper().fit(train)
    # Even if new data varies in "a", the fitted decision stands.
    new = pd.DataFrame({"a": [5, 6], "b": [1, 2]})
    assert list(dropper.transform(new).columns) == ["b"]
    assert dropper.dropped_columns_ == ["a"]


def test_imputer_medians_come_from_training_data_only():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [1.0, 2.0, 3.0, 4.0]})
    prep = build_imputing_preprocessor().fit(train)
    # Validation rows with absurd values must not influence the fitted median.
    val = pd.DataFrame({"a": [np.nan, 1e9], "b": [1e9, 1e9]})
    out = prep.transform(val)
    assert out[0, 0] == 2.0  # median of train column "a"


def test_missing_indicator_columns_are_added():
    train = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    out = build_imputing_preprocessor().fit_transform(train)
    assert out.shape[1] == 3  # 2 features + 1 indicator (only "a" had NaN)
    assert out[1, 2] == 1 and out[0, 2] == 0


def test_tree_preprocessor_keeps_nan():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "const": [7, 7, 7]})
    out = build_tree_preprocessor().fit_transform(X)
    assert list(out.columns) == ["a"]
    assert out["a"].isna().sum() == 1


def test_scaling_option_standardises():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20, 30, 40, 50]})
    out = build_imputing_preprocessor(scale=True).fit_transform(X)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-9)
