import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis, skew

from src.eda import distributions as dist


def test_column_distribution_matches_numpy_and_scipy():
    rng = np.random.default_rng(0)
    v = rng.lognormal(0, 1, 5000)
    d = dist.column_distribution(v)
    assert d["n_observed"] == 5000
    assert d["median"] == pytest.approx(np.median(v))
    assert d["q99"] == pytest.approx(np.quantile(v, 0.99))
    assert d["skew"] == pytest.approx(skew(v, bias=False))
    assert d["excess_kurtosis"] == pytest.approx(kurtosis(v, bias=False))
    assert d["max_over_q99"] == pytest.approx(v.max() / np.quantile(v, 0.99))


def test_outlier_share_exact_on_known_data():
    v = np.concatenate([np.arange(1, 100, dtype=float), [10_000.0]])   # 99 regular + 1 huge
    d = dist.column_distribution(v)
    q25, q75 = np.quantile(v, [0.25, 0.75])
    iqr = q75 - q25
    expected = np.mean((v < q25 - 1.5 * iqr) | (v > q75 + 1.5 * iqr))
    assert d["outlier_share_1_5iqr"] == pytest.approx(expected) == pytest.approx(0.01)
    assert d["outlier_share_3iqr"] == pytest.approx(0.01)


def test_zero_iqr_reports_nan_not_misleading_outliers():
    v = np.concatenate([np.zeros(90), np.arange(1, 11, dtype=float)])
    d = dist.column_distribution(v)
    assert d["iqr_is_zero"] is True
    assert np.isnan(d["outlier_share_1_5iqr"]) and np.isnan(d["outlier_share_3iqr"])
    assert d["zero_fraction_observed"] == pytest.approx(0.9)


def test_constant_column_has_nan_shape_stats():
    d = dist.column_distribution(np.full(50, 7.0))
    assert np.isnan(d["skew"]) and np.isnan(d["excess_kurtosis"]) and d["std"] == 0


def test_empty_column():
    assert dist.column_distribution([]) == {"n_observed": 0}


def _frame():
    rng = np.random.default_rng(1)
    n = 3000
    y = (rng.random(n) < 0.1).astype(int)
    y[:5] = 1
    heavy = rng.lognormal(0, 2.0, n)
    zeros = np.where(rng.random(n) < 0.7, 0.0, rng.poisson(5, n))
    X = pd.DataFrame(
        {
            "a_000": rng.normal(10, 1, n),
            "b_000": heavy,
            "c_000": zeros,
            "d_000": np.where(rng.random(n) < 0.2, np.nan, rng.normal(0, 1, n)),
            "e_000": np.nan,
            "f_000": y * 5.0 + rng.normal(0, 0.1, n),
        }
    )
    return X, y


def test_report_flags_heavy_tail_and_zero_inflation():
    X, y = _frame()
    r = dist.distribution_report(X, y)
    t = r["columns"]
    assert bool(t.loc["b_000", "heavy_tailed"]) and not bool(t.loc["a_000", "heavy_tailed"])
    assert bool(t.loc["c_000", "zero_inflated"]) and not bool(t.loc["a_000", "zero_inflated"])
    assert r["most_skewed_columns"][0] == "b_000"


def test_report_handles_missing_and_all_missing_columns():
    X, y = _frame()
    r = dist.distribution_report(X, y)
    t = r["columns"]
    assert t.loc["d_000", "n_observed"] == X["d_000"].notna().sum()
    assert t.loc["e_000", "n_observed"] == 0
    assert not bool(t.loc["e_000", "heavy_tailed"]) and not bool(t.loc["e_000", "zero_inflated"])
    assert r["n_columns_profiled"] == X.shape[1] - 1
    assert len(t) == X.shape[1]                      # every column stays in the table


def test_class_conditional_medians():
    X, y = _frame()
    t = dist.distribution_report(X, y)["columns"]
    assert t.loc["f_000", "median_pos"] == pytest.approx(5.0, abs=0.1)
    assert t.loc["f_000", "median_neg"] == pytest.approx(0.0, abs=0.1)


def test_report_is_descriptive_and_does_not_mutate():
    X, y = _frame()
    before = X.copy(deep=True)
    r = dist.distribution_report(X, y)
    pd.testing.assert_frame_equal(X, before)
    assert "No transformation" in r["policy"]
    assert not {"drop", "remove", "transform"} & set(r.keys())
