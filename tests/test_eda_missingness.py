import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

from src.eda import missingness as ms
from tests.eda_helpers import make_missingness_frame


@pytest.fixture(scope="module")
def frame():
    return make_missingness_frame()


@pytest.fixture(scope="module")
def report(frame):
    X, y = frame
    return ms.missingness_report(X, y, seed=1)


# ------------------------------ structure ---------------------------------- #
def test_indicator_matrix_drops_never_and_always_missing(frame):
    X, _ = frame
    M = ms.indicator_matrix(X)
    assert "full_000" not in M.columns and "none_000" not in M.columns
    assert M.dtypes.eq(bool).all()
    assert "inf_000" in M.columns


@pytest.mark.parametrize(
    "rate, bucket",
    [(0, "none"), (0.005, "(0,1%]"), (0.01, "(0,1%]"), (0.05, "(1,10%]"), (0.3, "(10,50%]"),
     (0.5, "(10,50%]"), (0.7, "(50,90%]"), (0.95, "(90,100%)"), (1.0, "all")],
)
def test_bucket_boundaries(rate, bucket):
    assert ms.missing_rate_bucket(rate) == bucket


def test_column_summary_counts(frame):
    X, y = frame
    s = ms.column_missing_summary(X, y)
    assert s.loc["none_000", "bucket"] == "all"
    assert s.loc["full_000", "bucket"] == "none"
    n_pos = int(y.sum())
    assert (s["n_pos_missing"] + s["n_pos_observed"] == n_pos).all()
    assert s.loc["inf_000", "n_missing"] == X["inf_000"].isna().sum()


def test_requires_both_classes(frame):
    X, _ = frame
    with pytest.raises(ValueError):
        ms.missingness_vs_target(X, np.zeros(len(X), dtype=int))
    with pytest.raises(ValueError):
        ms.row_missing_profile(X, np.ones(len(X), dtype=int))


# --------------------------- target association ----------------------------- #
def test_informative_column_flagged_null_column_not(report):
    cols = report["columns"]
    assert bool(cols.loc["inf_000", "significant_fdr"]) and bool(cols.loc["inf_000", "notable_effect"])
    assert cols.loc["inf_000", "indicator_auc"] > 0.8
    assert cols.loc["inf_000", "missing_lift"] > 5
    assert not bool(cols.loc["nul_000", "notable_effect"])
    assert abs(cols.loc["nul_000", "indicator_auc"] - 0.5) < 0.05


def test_indicator_auc_equals_sklearn(frame):
    X, y = frame
    table = ms.missingness_vs_target(X, y)
    for col in ["inf_000", "nul_000", "pa_000", "g_001"]:
        expected = roc_auc_score(y, X[col].isna().astype(int))
        assert table.loc[col, "indicator_auc"] == pytest.approx(expected, abs=1e-12)


def test_pvalues_match_scipy_fisher(frame):
    X, y = frame
    table = ms.missingness_vs_target(X, y)
    miss, pos = X["inf_000"].isna().to_numpy(), y.to_numpy() == 1
    a, b = int((miss & pos).sum()), int((miss & ~pos).sum())
    c, d = int(pos.sum()) - a, int((~pos).sum()) - b
    assert table.loc["inf_000", "p_value"] == pytest.approx(fisher_exact([[a, b], [c, d]])[1])
    assert table.loc["inf_000", "odds_ratio"] == pytest.approx((a * d) / (b * c))


def _manual_bh(p):
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def test_q_values_are_benjamini_hochberg(frame):
    X, y = frame
    table = ms.missingness_vs_target(X, y)
    assert np.allclose(table["q_value"].to_numpy(), _manual_bh(table["p_value"].to_numpy()))
    assert (table["q_value"] >= table["p_value"] - 1e-15).all()


def test_untested_columns_are_absent(frame):
    X, y = frame
    table = ms.missingness_vs_target(X, y)
    assert "full_000" not in table.index and "none_000" not in table.index


def test_odds_ratio_zero_cell_is_finite():
    assert np.isfinite(ms.odds_ratio(0, 10, 5, 20))
    assert ms.odds_ratio(0, 10, 5, 20) > 0
    assert ms.odds_ratio(10, 10, 10, 10) == pytest.approx(1.0)


def test_indicator_auc_formula_symmetry():
    assert ms.indicator_auc(0.5, 0.5) == 0.5
    assert ms.indicator_auc(0.8, 0.1) == pytest.approx(0.85)
    assert ms.indicator_auc(0.1, 0.8) == pytest.approx(0.15)


# ------------------------------ fold stability ------------------------------ #
def test_fold_stability_bounds_and_informative_column(frame):
    X, y = frame
    st = ms.fold_auc_stability(X, y, n_splits=5, seed=0)
    assert (st["auc_fold_min"] <= st["auc_fold_max"]).all()
    assert st.to_numpy().min() >= 0 and st.to_numpy().max() <= 1
    assert st.loc["inf_000", "auc_fold_min"] > 0.75        # stable, strong association
    assert st.loc["nul_000", "auc_fold_min"] < 0.6         # noise column wanders around 0.5


def test_fold_stability_is_seeded(frame):
    X, y = frame
    a = ms.fold_auc_stability(X, y, seed=3)
    b = ms.fold_auc_stability(X, y, seed=3)
    pd.testing.assert_frame_equal(a, b)


# -------------------------------- row level --------------------------------- #
def test_row_profile_matches_sklearn_and_direction(frame):
    X, y = frame
    prof = ms.row_missing_profile(X, y)
    counts = X.isna().sum(axis=1)
    assert prof["auc_missing_count_vs_target"] == pytest.approx(roc_auc_score(y, counts))
    assert prof["positive"]["mean"] > prof["negative"]["mean"]       # inf_000 pushes positives up
    assert prof["overall"]["quantiles"]["min"] >= 1                  # none_000 is always missing
    assert prof["overall"]["complete_row_share"] == 0.0              # none_000 => no row is complete
    q = prof["overall"]["quantiles"]
    assert q["min"] <= q["q25"] <= q["median"] <= q["q75"] <= q["q95"] <= q["max"]
    assert set(q) == {"min", "q25", "median", "q75", "q95", "max"}


# ------------------------------ co-missingness ------------------------------ #
def test_identical_pattern_groups(frame):
    X, _ = frame
    groups = ms.identical_pattern_groups(ms.indicator_matrix(X))
    assert ["pa_000", "pa_001"] in groups
    assert ["g_000", "g_001", "g_002", "g_003"] in groups
    assert not any("cl_000" in g for g in groups)        # flips make it non-identical
    assert not any("ind_000" in g for g in groups)


def test_clusters_find_planted_structure_and_ignore_noise(report):
    clusters = report["comissing_clusters"]
    as_sets = [set(c["columns"]) for c in clusters]
    assert {"g_000", "g_001", "g_002", "g_003"} in as_sets
    assert {"cl_000", "cl_001", "cl_002"} in as_sets
    assert {"pa_000", "pa_001"} in as_sets
    for noisy in ("ind_000", "ind_001", "nul_000", "inf_000"):
        assert not any(noisy in c["columns"] for c in clusters)
    assert [c["size"] for c in clusters] == sorted((c["size"] for c in clusters), reverse=True)
    assert all(c["min_pairwise_phi"] >= ms.COMISSING_MIN_PHI for c in clusters)


def test_clusters_are_deterministic(frame):
    X, _ = frame
    M = ms.indicator_matrix(X)
    assert ms.comissing_clusters(M) == ms.comissing_clusters(M)


def test_clusters_with_fewer_than_two_columns():
    X = pd.DataFrame({"a": [1.0, np.nan, 2.0, 3.0]})
    assert ms.comissing_clusters(ms.indicator_matrix(X)) == []
    assert ms.top_comissing_pairs(ms.indicator_matrix(X)) == []


def test_top_pairs_sorted_and_correct(frame):
    X, _ = frame
    M = ms.indicator_matrix(X)
    pairs = ms.top_comissing_pairs(M, k=5)
    phis = [p["phi"] for p in pairs]
    assert phis == sorted(phis, reverse=True)
    assert pairs[0]["phi"] == pytest.approx(1.0)
    a, b = pairs[0]["a"], pairs[0]["b"]
    assert pairs[0]["phi"] == pytest.approx(np.corrcoef(M[a].astype(float), M[b].astype(float))[0, 1])


# ------------------------------ group structure ------------------------------ #
def test_group_structure(frame):
    X, y = frame
    groups = ms.group_missingness_structure(X, y)
    g = groups["g"]
    assert g["n_features"] == 4
    assert g["partial_missing_share"] == 0.0                         # whole group goes missing together
    assert g["all_missing_share"] == pytest.approx(X["g_000"].isna().mean())
    assert g["all_missing_share"] + g["complete_share"] == pytest.approx(1.0)
    assert groups["cl"]["partial_missing_share"] > 0                 # flips -> partial states exist
    assert "inf" not in groups                                       # single-column groups skipped


# -------------------------------- row patterns ------------------------------- #
def test_row_patterns_account_for_every_row(frame):
    X, y = frame
    M = ms.indicator_matrix(X)
    full = ms.row_pattern_summary(M, y, top_k=10_000)
    assert sum(p["n_rows"] for p in full["top_patterns"]) == len(X)
    assert sum(p["positives"] for p in full["top_patterns"]) == int(y.sum())
    assert full["top_k_coverage"] == pytest.approx(1.0)
    assert full["n_distinct_patterns"] == len(full["top_patterns"])


def test_row_patterns_match_a_pandas_groupby(frame):
    X, y = frame
    M = ms.indicator_matrix(X)
    expected = M.groupby(list(M.columns)).size().sort_values(ascending=False)
    top = ms.row_pattern_summary(M, y, top_k=3)["top_patterns"]
    assert [p["n_rows"] for p in top] == expected.head(3).tolist()
    assert top[0]["positive_rate"] == top[0]["positives"] / top[0]["n_rows"]


# --------------------------------- aggregate -------------------------------- #
def test_report_policy_and_no_removal_language(report):
    assert "No feature is removed" in report["policy"]
    forbidden = {"drop", "remove", "removed", "to_drop", "recommended_removal"}
    assert not forbidden & set(report.keys())
    assert not forbidden & set(report["columns"].columns)


def test_report_does_not_mutate_input(frame):
    X, y = frame
    before = X.copy(deep=True)
    ms.missingness_report(X, y)
    pd.testing.assert_frame_equal(X, before)


def test_report_counts_are_consistent(report, frame):
    X, _ = frame
    assert report["n_columns_total"] == X.shape[1]
    assert report["n_columns_all_missing"] == 1
    assert sum(report["bucket_counts"].values()) == X.shape[1]
    ta = report["target_association"]
    assert ta["n_tested"] == report["n_columns_with_missing"] - report["n_columns_all_missing"]
    assert ta["n_higher_missing_in_positives"] + ta["n_lower_missing_in_positives"] <= ta["n_tested"]
    assert ta["strongest_columns"][0] == "inf_000"
    assert len(report["columns"]) == X.shape[1]                       # every column kept in the table


# --------------------------------- edge cases -------------------------------- #
def test_report_on_data_with_no_missing_values():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a_000": rng.normal(size=200), "a_001": rng.normal(size=200)})
    y = pd.Series((rng.random(200) < 0.1).astype(int))
    y.iloc[:3] = 1
    r = ms.missingness_report(X, y)
    assert r["n_columns_with_missing"] == 0
    assert r["target_association"]["n_tested"] == 0
    assert r["comissing_clusters"] == [] and r["row_patterns"]["n_distinct_patterns"] == 1


def test_report_with_a_single_varying_column():
    rng = np.random.default_rng(1)
    v = rng.normal(size=300)
    v[rng.random(300) < 0.3] = np.nan
    X = pd.DataFrame({"a_000": v, "b_000": rng.normal(size=300)})
    y = pd.Series((rng.random(300) < 0.1).astype(int))
    y.iloc[:5] = 1
    r = ms.missingness_report(X, y)
    assert r["target_association"]["n_tested"] == 1
    assert r["comissing_clusters"] == []


def test_runner_section_output_is_json_clean_and_complete(tmp_path):
    import json
    from src.eda import run_eda as runner
    from tests.conftest import write_aps_csv

    X, y = make_missingness_frame(n=1500, seed=2)
    df = X.copy()
    df.insert(0, "class", np.where(y == 1, "pos", "neg"))
    csv = write_aps_csv(df, tmp_path / "train.csv")

    results = runner.run_eda(csv, tmp_path / "o", seed=4, sections=["missingness"])
    saved = json.loads((tmp_path / "o" / "eda_results.json").read_text())
    section = saved["sections"]["missingness"]

    assert len(section["columns"]) == X.shape[1]
    assert section["columns"]["inf_000"]["significant_fdr"] is True
    assert "No feature is removed" in section["policy"]
    text = runner.format_summary(results)
    assert "target association" in text and "co-missing clusters" in text


# ------------- 2d-1: significant / notable direction splits (additive fields) ------------- #
NEW_KEYS = (
    "n_significant_higher_missing_in_positives",
    "n_significant_lower_missing_in_positives",
    "n_notable_higher_missing_in_positives",
    "n_notable_lower_missing_in_positives",
    "n_significant_and_notable",
)


@pytest.fixture(scope="module")
def two_direction_report():
    X, y = make_missingness_frame()
    rng = np.random.default_rng(21)
    pos = y.to_numpy() == 1
    low = rng.normal(size=len(X))
    low[rng.random(len(X)) < np.where(pos, 0.05, 0.40)] = np.nan     # LESS missing in failures
    X = X.assign(low_000=low)
    return X, y, ms.missingness_report(X, y, seed=1)


def test_new_fields_match_manual_counts(two_direction_report):
    X, y, report = two_direction_report
    cols = report["columns"].dropna(subset=["indicator_auc"])
    sig, notable = cols["significant_fdr"].astype(bool), cols["notable_effect"].astype(bool)
    higher, lower = cols["indicator_auc"] > 0.5, cols["indicator_auc"] < 0.5
    ta = report["target_association"]
    assert ta["n_significant_higher_missing_in_positives"] == int((sig & higher).sum())
    assert ta["n_significant_lower_missing_in_positives"] == int((sig & lower).sum())
    assert ta["n_notable_higher_missing_in_positives"] == int((notable & higher).sum())
    assert ta["n_notable_lower_missing_in_positives"] == int((notable & lower).sum())
    assert ta["n_significant_and_notable"] == int((sig & notable).sum())


def test_planted_lower_missing_column_lands_in_the_lower_bucket(two_direction_report):
    _, _, report = two_direction_report
    row = report["columns"].loc["low_000"]
    assert row["indicator_auc"] < 0.5 and bool(row["significant_fdr"]) and bool(row["notable_effect"])
    ta = report["target_association"]
    assert ta["n_significant_lower_missing_in_positives"] >= 1       # low_000
    assert ta["n_significant_higher_missing_in_positives"] >= 1      # inf_000 (planted earlier)


def test_splits_partition_the_existing_totals(two_direction_report):
    _, _, report = two_direction_report
    ta = report["target_association"]
    assert ta["n_significant_higher_missing_in_positives"] + ta["n_significant_lower_missing_in_positives"] \
        == ta["n_significant_fdr"]
    assert ta["n_notable_higher_missing_in_positives"] + ta["n_notable_lower_missing_in_positives"] \
        == ta["n_notable_effect"]
    assert ta["n_significant_and_notable"] <= min(ta["n_significant_fdr"], ta["n_notable_effect"])


def test_existing_target_association_fields_keep_their_definition(two_direction_report):
    X, y, report = two_direction_report
    tested = ms.missingness_vs_target(X, y)
    ta = report["target_association"]
    assert ta["n_higher_missing_in_positives"] == int((tested["indicator_auc"] > 0.5).sum())
    assert ta["n_lower_missing_in_positives"] == int((tested["indicator_auc"] < 0.5).sum())
    assert ta["n_significant_fdr"] == int(tested["significant_fdr"].sum())
    assert ta["n_notable_effect"] == int(tested["notable_effect"].sum())
    for key in ("alpha", "notable_auc_delta", "n_tested", "strongest_columns"):
        assert key in ta


def test_new_fields_are_zero_when_nothing_is_tested():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a_000": rng.normal(size=200), "a_001": rng.normal(size=200)})
    y = pd.Series((rng.random(200) < 0.1).astype(int))
    y.iloc[:3] = 1
    ta = ms.missingness_report(X, y)["target_association"]
    assert all(ta[k] == 0 for k in NEW_KEYS)


def test_new_fields_are_plain_ints_for_json(two_direction_report):
    ta = two_direction_report[2]["target_association"]
    assert all(type(ta[k]) is int for k in NEW_KEYS)


def _asymmetric_frame():
    """Counts are deliberately all different so a swapped direction or a wrong set operation shows.

    columns (missing rate in failures vs non-failures; n=30,000, 10% failures):
      h1..h3  0.60 vs 0.10  -> significant, notable, HIGHER
      l1      0.10 vs 0.50  -> significant, notable, LOWER
      su1,su2 0.33 vs 0.30  -> significant, NOT notable, HIGHER
      sd1     0.27 vs 0.30  -> significant, NOT notable, LOWER
      n1..n3  0.30 vs 0.30  -> null
    expected: sig higher 5, sig lower 2, notable higher 3, notable lower 1, significant-and-notable 4
    """
    rng = np.random.default_rng(99)
    n = 30_000
    y = (rng.random(n) < 0.10).astype(int)
    pos = y == 1
    design = {"h1": (0.60, 0.10), "h2": (0.60, 0.10), "h3": (0.60, 0.10), "l1": (0.10, 0.50),
              "su1": (0.33, 0.30), "su2": (0.33, 0.30), "sd1": (0.27, 0.30),
              "n1": (0.30, 0.30), "n2": (0.30, 0.30), "n3": (0.30, 0.30)}
    data = {}
    for k, (name, (p_pos, p_neg)) in enumerate(design.items()):
        values = rng.normal(size=n)
        values[rng.random(n) < np.where(pos, p_pos, p_neg)] = np.nan
        data[f"{name[:-1] if name[-1].isdigit() else name}_{k:03d}"] = values
    return pd.DataFrame(data), pd.Series(y), design


def test_direction_splits_on_an_asymmetric_design():
    X, y, _ = _asymmetric_frame()
    ta = ms.missingness_report(X, y)["target_association"]
    assert ta["n_significant_higher_missing_in_positives"] == 5
    assert ta["n_significant_lower_missing_in_positives"] == 2
    assert ta["n_notable_higher_missing_in_positives"] == 3
    assert ta["n_notable_lower_missing_in_positives"] == 1
    assert ta["n_significant_and_notable"] == 4
    assert ta["n_significant_fdr"] == 7 and ta["n_notable_effect"] == 4      # existing fields agree
