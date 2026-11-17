import math

import numpy as np
import pandas as pd
import pytest

from src.eda import quality
from tests.eda_helpers import make_eda_frame


@pytest.fixture(scope="module")
def frame():
    return make_eda_frame()


def test_constant_columns(frame):
    X, y = frame
    # Milestone 1's helper counts an all-NaN column as constant (nunique <= 1).
    assert quality.quality_report(X, y)["constant_columns"] == ["ae_000", "cd_000"]


def test_all_missing_column_found(frame):
    X, _ = frame
    assert quality.all_missing_columns(X) == ["ae_000"]


def test_near_constant_includes_planted_column(frame):
    X, _ = frame
    near = quality.near_constant_columns(X)
    assert "ca_000" in near and near["ca_000"] > 0.999
    assert "cd_000" in near
    assert "aa_000" not in near


def test_duplicate_columns_detected_and_confirmed(frame):
    X, _ = frame
    assert quality.duplicate_columns(X) == [["aa_000", "ad_000"]]


def test_duplicate_columns_respect_nan_positions():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0], "c": [1.0, np.nan, 3.0]})
    assert quality.duplicate_columns(X) == [["a", "c"]]


def test_no_duplicates_returns_empty():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    assert quality.duplicate_columns(X) == []


def test_negative_and_non_integer_columns(frame):
    X, y = frame
    report = quality.quality_report(X, y)
    assert "af_000" in report["columns_with_negative_values"]
    assert "ac_000" in report["columns_with_negative_values"]
    assert "ac_000" in report["non_integer_columns"]
    assert "af_000" not in report["non_integer_columns"]
    assert "aa_000" not in report["non_integer_columns"]


def test_saturation_candidate_flagged_but_not_low_cardinality():
    X, y = make_eda_frame()
    report = quality.quality_report(X, y)
    assert "ai_000" in report["saturation_candidates"]
    # a binary column has a "repeated max" naturally and must not be flagged
    Xb = pd.DataFrame({"flag": np.random.default_rng(0).integers(0, 2, 500).astype(float)})
    assert quality.quality_report(Xb, np.zeros(500, dtype=int))["saturation_candidates"] == {}


def test_non_numeric_columns_reported():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    assert quality.non_numeric_columns(X) == ["b"]


def test_value_profile_facts(frame):
    X, _ = frame
    profile = quality.value_profile(X)
    assert profile.loc["ab_000", "missing_rate"] == pytest.approx(X["ab_000"].isna().mean())
    assert profile.loc["cd_000", "n_unique"] == 1
    assert bool(profile.loc["aa_000", "integer_valued"]) is True
    assert profile.loc["ae_000", "n_missing"] == len(X)
    assert profile.loc["aa_000", "zero_fraction"] >= 0


def test_sparse_columns_are_never_flagged_for_removal(frame):
    X, y = frame
    report = quality.quality_report(X, y)
    # ab_000 is ~30% missing: it must appear nowhere as a "bad" column.
    for key in ("constant_columns", "all_missing_columns", "duplicate_column_groups"):
        assert "ab_000" not in str(report[key])
    assert "ab_000" not in report["near_constant_columns"]


def test_wilson_interval_matches_known_value():
    from scipy.stats import binomtest

    low, high = quality.wilson_interval(1000, 60000)
    ref = binomtest(1000, 60000).proportion_ci(confidence_level=0.95, method="wilson")
    REF_LOW, REF_HIGH = ref.low, ref.high
    # cross-checked against scipy.stats.binomtest(...).proportion_ci(method="wilson")
    assert low == pytest.approx(REF_LOW, abs=1e-6)
    assert high == pytest.approx(REF_HIGH, abs=1e-6)
    assert low < 1000 / 60000 < high


def test_wilson_interval_edge_cases():
    assert math.isnan(quality.wilson_interval(0, 0)[0])
    low, high = quality.wilson_interval(0, 100)
    assert low == pytest.approx(0.0, abs=1e-12) and high > 0


def test_class_balance():
    y = np.array([0] * 98 + [1] * 2)
    cb = quality.class_balance(y)
    assert (cb["n"], cb["negatives"], cb["positives"]) == (100, 98, 2)
    assert cb["imbalance_ratio"] == pytest.approx(49.0)
    assert cb["positive_rate"] == pytest.approx(0.02)


def test_class_balance_no_positives():
    cb = quality.class_balance(np.zeros(10, dtype=int))
    assert cb["positives"] == 0 and math.isnan(cb["imbalance_ratio"])
