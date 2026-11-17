import numpy as np
import pandas as pd

from src.data.audit import audit_training_data, constant_columns, missingness_report


def test_constant_columns_detected(xy):
    X, _ = xy
    assert constant_columns(X) == ["cd_000"]


def test_all_nan_column_counts_as_constant():
    X = pd.DataFrame({"a": [np.nan, np.nan], "b": [1, 2]})
    assert constant_columns(X) == ["a"]


def test_missingness_lift_flags_informative_missingness(xy):
    X, y = xy
    report = missingness_report(X, y)
    assert report.loc["f_3", "missing_lift"] > 1.5


def test_audit_summary(xy):
    X, y = xy
    summary = audit_training_data(X, y)
    assert summary["rows"] == len(X)
    assert summary["positives"] == int(y.sum())
    assert "cd_000" in summary["constant_columns"]
