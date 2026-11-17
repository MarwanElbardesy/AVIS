import numpy as np
import pandas as pd
import pytest

from src.eda import leakage as lk
from src.eda import signal as sg


def _clean(n=6000, seed=0, pos_rate=0.05):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    X = pd.DataFrame(
        {
            "a_000": rng.normal(0, 1, n) + 0.3 * y,
            "b_000": rng.poisson(10, n).astype(float),
            "c_000": np.where(rng.random(n) < 0.2, np.nan, rng.normal(0, 1, n)),
        }
    )
    return X, y


def _report(X, y):
    return lk.leakage_report(X, y, sg.signal_report(X, y, seed=0)["columns"])


# --------------------------------- row order ---------------------------------- #
def test_random_order_is_not_flagged():
    X, y = _clean()
    r = _report(X, y)
    assert not any(r["flags"].values()), r["flags"]
    assert r["row_order"]["blocks"]["p_value"] > 0.01


def test_sorted_labels_are_flagged_by_blocks_and_runs():
    X, y = _clean()
    order = np.argsort(y, kind="mergesort")                  # all failures at the end of the file
    Xs, ys = X.iloc[order].reset_index(drop=True), y[order]
    r = _report(Xs, ys)
    assert r["flags"]["row_order_related_to_label"] is True
    assert r["row_order"]["blocks"]["cramers_v"] > 0.5
    assert r["row_order"]["runs_test"]["z"] < -10


def test_runs_test_matches_hand_computation():
    r = lk.runs_test([0, 0, 1, 1, 0])
    # n1=2, n0=3, n=5, runs=3: expected=3.4, var=0.84, z=-0.4/sqrt(0.84)
    assert r["runs"] == 3 and r["expected_runs"] == pytest.approx(3.4)
    assert r["z"] == pytest.approx(-0.4 / np.sqrt(0.84))


def test_alternating_labels_give_positive_z():
    assert lk.runs_test(np.tile([0, 1], 500))["z"] > 10


def test_block_rates_are_per_block_fractions():
    y = np.zeros(1000, dtype=int)
    y[:100] = 1                                              # all in the first block
    b = lk.block_positive_rates(y, n_blocks=10)
    assert b["positive_rate_by_block"][0] == 1.0 and sum(b["positive_rate_by_block"][1:]) == 0.0
    assert b["p_value"] < 1e-10


# --------------------------------- drift -------------------------------------- #
def test_feature_drifting_with_row_position_is_flagged():
    X, y = _clean()
    X["t_000"] = np.arange(len(X)) + np.random.default_rng(1).normal(0, 50, len(X))
    r = _report(X, y)
    assert r["flags"]["features_drift_with_row_position"] is True
    d = r["feature_drift"]
    assert d["top_drifting_columns"][0]["column"] == "t_000" and d["max_abs_rho"] > 0.95


def test_missing_count_drift_is_reported():
    X, y = _clean()
    n = len(X)
    X.loc[: n // 2, "b_000"] = np.nan                         # first half has more missing values
    assert lk.feature_index_drift(X)["row_missing_count_vs_position_rho"] < -0.3


# ----------------------- suspiciously strong / exclusive ---------------------- #
def test_near_perfect_feature_is_flagged():
    X, y = _clean()
    X["leak_000"] = y + np.random.default_rng(2).normal(0, 0.01, len(y))
    r = _report(X, y)
    assert r["flags"]["suspiciously_strong_single_feature"] is True
    assert r["suspicious_single_features"]["value_auc"] == ["leak_000"]


def test_missingness_that_encodes_the_label_is_flagged():
    X, y = _clean()
    X.loc[y == 1, "c_000"] = np.nan                           # missing exactly when failure
    X.loc[y == 0, "c_000"] = X.loc[y == 0, "c_000"].fillna(0.0)
    r = _report(X, y)
    assert "c_000" in r["suspicious_single_features"]["indicator_auc"]


def test_class_exclusive_values_found_only_with_enough_rows():
    X, y = _clean()
    rng = np.random.default_rng(3)
    shared = rng.integers(0, 3, len(y)).astype(float)                     # 0,1,2 occur in both classes
    code = np.where((y == 1) & (rng.random(len(y)) < 0.5), 9.0, shared)  # 9.0 occurs only among failures
    X["cat_000"] = code
    found = lk.class_exclusive_values(X, y)
    assert [e["value"] for e in found["cat_000"]] == [9.0]               # shared values are NOT reported
    assert found["cat_000"][0]["only_class"] == 1
    assert found["cat_000"][0]["count"] == int((code == 9.0).sum())
    # too few rows with that value -> not reported
    assert lk.class_exclusive_values(X, y, min_count=10_000) == {}
    # high-cardinality columns are skipped entirely
    assert "a_000" not in found


def test_class_exclusive_values_detects_both_directions():
    X, y = _clean()
    X["cat_000"] = np.where(y == 1, 1.0, np.random.default_rng(4).integers(2, 5, len(y)).astype(float))
    found = {e["value"]: e["only_class"] for e in lk.class_exclusive_values(X, y)["cat_000"]}
    assert found[1.0] == 1 and found[2.0] == 0 and found[3.0] == 0 and found[4.0] == 0


# --------------------------------- duplicates --------------------------------- #
def test_duplicate_rows_with_conflicting_labels():
    X, y = _clean(n=200)
    X.iloc[1] = X.iloc[0]
    y = y.copy()
    y[0], y[1] = 0, 1
    d = lk.duplicate_row_check(X, y)
    assert d["n_rows_in_duplicate_groups"] == 2 and d["n_conflicting_label_groups"] == 1
    assert _report(X, y)["flags"]["duplicate_rows_with_conflicting_labels"] is True


def test_no_duplicates():
    X, y = _clean(n=200)
    assert lk.duplicate_row_check(X, y) == {"n_rows_in_duplicate_groups": 0, "n_conflicting_label_groups": 0}


# --------------------------------- reporting ---------------------------------- #
def test_policy_states_the_vehicle_id_limitation():
    r = _report(*_clean(n=500))
    assert "vehicle id" in r["policy"] and "review prompts" in r["policy"]
    assert set(r["flags"]) == {
        "row_order_related_to_label",
        "features_drift_with_row_position",
        "suspiciously_strong_single_feature",
        "class_exclusive_values_present",
        "duplicate_rows_with_conflicting_labels",
    }


def test_report_is_deterministic_and_does_not_mutate():
    X, y = _clean(n=800)
    before = X.copy(deep=True)
    assert _report(X, y) == _report(X, y)
    pd.testing.assert_frame_equal(X, before)
