import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.eda import missingness as ms
from src.eda import signal as sg


def test_mann_whitney_auc_equals_sklearn_including_ties():
    rng = np.random.default_rng(0)
    labels = rng.random(500) < 0.2
    values = np.round(rng.normal(0, 1, 500) + labels, 1)          # rounding => many ties
    auc, n1, n0 = sg.mann_whitney_auc(values, labels)
    assert auc == pytest.approx(roc_auc_score(labels, values), abs=1e-12)
    assert (n1, n0) == (int(labels.sum()), int((~labels).sum()))


def test_auc_single_class_is_nan():
    auc, n1, n0 = sg.mann_whitney_auc([1.0, 2.0, 3.0], [False, False, False])
    assert np.isnan(auc) and (n1, n0) == (0, 3)


def test_hanley_mcneil_se_matches_monte_carlo():
    rng = np.random.default_rng(42)
    n1, n0, aucs = 40, 160, []
    for _ in range(400):
        pos, neg = rng.normal(1.0, 1, n1), rng.normal(0, 1, n0)
        aucs.append(sg.mann_whitney_auc(np.concatenate([pos, neg]), np.r_[np.ones(n1), np.zeros(n0)] == 1)[0])
    empirical_sd = np.std(aucs, ddof=1)
    theoretical = sg.hanley_mcneil_se(np.mean(aucs), n1, n0)
    assert theoretical == pytest.approx(empirical_sd, rel=0.2)


def test_hanley_mcneil_edge_cases():
    assert np.isnan(sg.hanley_mcneil_se(float("nan"), 10, 10))
    assert np.isnan(sg.hanley_mcneil_se(0.7, 0, 10))
    assert sg.hanley_mcneil_se(0.5, 100, 100) > sg.hanley_mcneil_se(0.5, 1000, 1000)


@pytest.fixture(scope="module")
def planted():
    rng = np.random.default_rng(5)
    n = 4000
    y = (rng.random(n) < 0.08).astype(int)
    y[:40] = 1
    pos = y == 1

    sparse = np.full(n, np.nan)
    keep = np.flatnonzero(~pos)[:100].tolist() + np.flatnonzero(pos)[:5].tolist()
    sparse[keep] = rng.normal(0, 1, len(keep))

    X = pd.DataFrame(
        {
            "sig_000": rng.normal(0, 1, n) + 2.0 * y,                          # higher in failures
            "sig_001": rng.normal(0, 1, n) - 2.0 * y,                          # lower in failures
            "noi_000": rng.normal(0, 1, n),                                    # pure noise
            "low_000": sparse,                                                 # 5 observed failures
            "con_000": 3.0,
            "nan_000": np.nan,
            "ind_000": np.where(rng.random(n) < np.where(pos, 0.9, 0.05), np.nan, rng.normal(0, 1, n)),
        }
    )
    return X, y


@pytest.fixture(scope="module")
def result(planted):
    X, y = planted
    return sg.signal_report(X, y, seed=1)


def test_direction_and_orientation(result):
    t = result["columns"]
    assert t.loc["sig_000", "direction"] == "higher_in_failures" and t.loc["sig_000", "auc"] > 0.8
    assert t.loc["sig_001", "direction"] == "lower_in_failures" and t.loc["sig_001", "auc"] < 0.2
    assert t.loc["sig_001", "auc_oriented"] == pytest.approx(1 - t.loc["sig_001", "auc"])
    assert (t["auc_oriented"].dropna() >= 0.5).all()


def test_noise_is_near_chance(result):
    assert result["columns"].loc["noi_000", "auc_oriented"] < 0.56


def test_matches_sklearn_on_observed_rows(planted, result):
    X, y = planted
    ok = X["ind_000"].notna().to_numpy()
    expected = roc_auc_score(y[ok], X.loc[ok, "ind_000"])
    assert result["columns"].loc["ind_000", "auc"] == pytest.approx(expected, abs=1e-12)


def test_unscorable_columns_are_kept_but_flagged(result):
    t = result["columns"]
    assert np.isnan(t.loc["con_000", "auc"]) and np.isnan(t.loc["nan_000", "auc"])
    assert t.loc["con_000", "direction"] == "none"
    assert result["n_columns_not_scorable"] == 2
    assert result["n_columns_scored"] + result["n_columns_not_scorable"] == len(t) == 7


def test_low_support_flag(result):
    t = result["columns"]
    assert t.loc["low_000", "n_pos_obs"] == 5 and bool(t.loc["low_000", "low_support"])
    assert not bool(t.loc["sig_000", "low_support"])
    assert result["n_low_support"] >= 1


def test_indicator_auc_agrees_with_missingness_module(planted, result):
    X, y = planted
    table = ms.missingness_vs_target(X, y)
    for col in table.index:
        assert result["columns"].loc[col, "indicator_auc"] == pytest.approx(table.loc[col, "indicator_auc"])
    assert result["columns"].loc["ind_000", "indicator_auc_oriented"] < 0.2 or \
        result["columns"].loc["ind_000", "indicator_auc_oriented"] > 0.8


def test_fold_stability_fields(result):
    t = result["columns"]
    assert (t["auc_fold_min"].dropna() <= t["auc_fold_max"].dropna()).all()
    assert bool(t.loc["sig_000", "fold_sign_consistent"]) and bool(t.loc["sig_001", "fold_sign_consistent"])
    assert t.loc["sig_000", "auc_fold_min"] > 0.7
    assert not bool(t.loc["con_000", "fold_sign_consistent"])


def test_summary_structure(result):
    assert sum(result["auc_band_counts"].values()) == result["n_columns_scored"]
    top = [c["column"] for c in result["top_columns"]]
    assert top[:2] == ["sig_000", "sig_001"] or set(top[:2]) == {"sig_000", "sig_001"}
    assert result["by_group"]["sig"]["n_features"] == 2 and result["by_group"]["sig"]["n_ge_0_7"] == 2


def test_policy_warns_against_selection(result):
    assert "leak" in result["policy"] and "inside CV folds" in result["policy"]
    assert not {"selected_features", "drop", "remove"} & set(result)


def test_seeded_and_input_not_mutated(planted):
    X, y = planted
    before = X.copy(deep=True)
    a, b = sg.signal_report(X, y, seed=9), sg.signal_report(X, y, seed=9)
    pd.testing.assert_frame_equal(a["columns"], b["columns"])
    pd.testing.assert_frame_equal(X, before)


def test_requires_both_classes(planted):
    X, _ = planted
    with pytest.raises(ValueError):
        sg.signal_report(X, np.zeros(len(X), dtype=int))
