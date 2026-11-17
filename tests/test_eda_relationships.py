import numpy as np
import pandas as pd
import pytest

from src.eda import relationships as rel


def _corr_frame(n=3000, seed=0, missing=False):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a_000": a,
            "b_000": 2 * a + rng.normal(0, 0.05, n),          # near duplicate of a
            "c_000": -a + rng.normal(0, 0.05, n),             # negative near duplicate
            "d_000": rng.normal(size=n),                      # independent
            "e_000": np.exp(a) + rng.normal(0, 0.5, n),       # monotone, nonlinear
            "k_000": 5.0,                                     # constant
        }
    )
    if missing:
        for col in ("a_000", "d_000", "e_000"):
            X.loc[rng.random(n) < 0.3, col] = np.nan
    return X


def test_equals_spearman_when_nothing_is_missing():
    X = _corr_frame().drop(columns="k_000")
    ours = rel.pairwise_rank_correlation(X)
    expected = X.corr(method="spearman")
    assert np.allclose(ours.to_numpy(), expected.to_numpy(), atol=1e-10)


def test_close_to_pairwise_spearman_with_missing_values():
    X = _corr_frame(missing=True).drop(columns="k_000")
    ours = rel.pairwise_rank_correlation(X)
    expected = X.corr(method="spearman")           # pandas: pairwise-complete
    assert np.nanmax(np.abs(ours.to_numpy() - expected.to_numpy())) < 0.02


def test_constant_column_and_small_overlap_are_nan():
    X = _corr_frame()
    rho = rel.pairwise_rank_correlation(X)
    assert rho["k_000"].isna().all() and rho.loc["k_000"].isna().all()
    assert rho.loc["a_000", "a_000"] == pytest.approx(1.0)

    sparse = X[["a_000", "b_000"]].copy()
    sparse.iloc[50:, 0] = np.nan                    # only 50 shared rows
    assert np.isnan(rel.pairwise_rank_correlation(sparse, min_overlap=100).loc["a_000", "b_000"])


def test_matrix_is_symmetric_and_bounded():
    rho = rel.pairwise_rank_correlation(_corr_frame(missing=True)).to_numpy()
    assert np.allclose(rho, rho.T, equal_nan=True)
    assert np.nanmax(np.abs(rho)) <= 1.0


def test_all_missing_column_does_not_break_or_warn(recwarn):
    X = _corr_frame()
    X["z_000"] = np.nan
    rho = rel.pairwise_rank_correlation(X)
    assert rho["z_000"].isna().all()
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_near_duplicate_clusters_include_negative_correlation():
    X = _corr_frame()
    clusters = rel.near_duplicate_clusters(rel.pairwise_rank_correlation(X))
    assert len(clusters) == 1
    assert set(clusters[0]["columns"]) == {"a_000", "b_000", "c_000"}
    assert clusters[0]["min_pairwise_abs_rho"] >= rel.NEAR_DUPLICATE_RHO - 0.02


def test_clusters_empty_when_independent():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(2000, 5)), columns=[f"x_{i:03d}" for i in range(5)])
    assert rel.near_duplicate_clusters(rel.pairwise_rank_correlation(X)) == []


def test_top_pairs_sorted_by_abs_rho():
    pairs = rel.top_correlated_pairs(rel.pairwise_rank_correlation(_corr_frame()), k=4)
    strengths = [abs(p["rho"]) for p in pairs]
    assert strengths == sorted(strengths, reverse=True)
    assert {pairs[0]["a"], pairs[0]["b"]} <= {"a_000", "b_000", "c_000"}


def _hist_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    fixed = rng.multinomial(100, np.linspace(1, 10, 10) / 55, size=n)        # constant row total = 100
    for b in range(10):
        data[f"ag_{b:03d}"] = fixed[:, b].astype(float)
    for b in range(10):
        data[f"ay_{b:03d}"] = rng.poisson(5, n).astype(float)               # varying total
    for b in range(3):
        data[f"bb_{b:03d}"] = rng.poisson(5, n).astype(float)               # not a histogram candidate
    X = pd.DataFrame(data)
    X.loc[:199, "ay_003"] = np.nan                                           # 200 incomplete rows
    return X


def test_histogram_verification_distinguishes_constant_and_varying_totals():
    X = _hist_frame()
    rho = rel.pairwise_rank_correlation(X)
    h = rel.verify_histogram_groups(X, rho)
    assert set(h) == {"ag", "ay"}                                            # bb is not a candidate
    assert h["ag"]["total_is_constant"] is True and h["ag"]["row_total_cv"] == 0.0
    assert h["ag"]["row_total_min"] == h["ag"]["row_total_max"] == 100.0
    assert h["ay"]["total_is_constant"] is False and h["ay"]["row_total_cv"] > 0
    assert h["ay"]["n_complete_rows"] == 1800
    assert h["ay"]["complete_share"] == pytest.approx(0.9)
    assert h["ag"]["any_negative_values"] is False


def test_histogram_negative_values_and_all_zero_share():
    X = _hist_frame()
    X.loc[0, "ag_000"] = -1.0
    X.loc[1:20, [f"ag_{b:03d}" for b in range(10)]] = 0.0
    h = rel.verify_histogram_groups(X, rel.pairwise_rank_correlation(X))
    assert h["ag"]["any_negative_values"] is True
    assert h["ag"]["all_zero_share"] == pytest.approx(20 / 2000)


def test_relationships_report_is_descriptive_and_deterministic():
    X = _hist_frame()
    before = X.copy(deep=True)
    a, b = rel.relationships_report(X), rel.relationships_report(X)
    pd.testing.assert_frame_equal(X, before)
    assert a == b
    assert "never dropped" in a["policy"]
    assert not {"drop", "remove", "to_drop"} & set(a)


def test_correct_under_informative_missingness():
    """Each column goes missing in a different region of a shared driver, so a column's own
    observed rows differ from the pair's overlap. The overlap-mean correction in the covariance
    is what keeps the result right; without it the estimate is off by ~0.09 here."""
    rng = np.random.default_rng(0)
    t = rng.normal(size=6000)
    a = np.where(t > 1, np.nan, t + rng.normal(0, 0.7, 6000))
    b = np.where(t < -1, np.nan, t + rng.normal(0, 0.7, 6000))
    X = pd.DataFrame({"a_000": a, "b_000": b, "c_000": rng.normal(size=6000)})
    ours = rel.pairwise_rank_correlation(X).loc["a_000", "b_000"]
    reference = X.corr(method="spearman").loc["a_000", "b_000"]
    assert ours == pytest.approx(reference, abs=2e-3)


# ------------------- 2d-1: shared-row counts (additive fields) -------------------- #
def _overlap_frame():
    """a/b/c correlated; a is observed in fewer rows, so pairs involving it have less overlap."""
    rng = np.random.default_rng(11)
    n = 3000
    t = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a_000": t + rng.normal(0, 0.05, n),
            "b_000": t + rng.normal(0, 0.05, n),
            "c_000": t + rng.normal(0, 0.05, n),
            "d_000": rng.normal(size=n),
            "z_000": np.nan,
        }
    )
    X.loc[rng.random(n) < 0.6, "a_000"] = np.nan          # a observed in ~40% of rows
    X.loc[rng.random(n) < 0.1, "b_000"] = np.nan
    return X


def test_overlap_counts_match_manual_counts():
    X = _overlap_frame()
    ov = rel.pairwise_overlap_counts(X)
    for a, b in [("a_000", "b_000"), ("a_000", "c_000"), ("b_000", "c_000"), ("a_000", "d_000")]:
        assert ov.loc[a, b] == int((X[a].notna() & X[b].notna()).sum())
    assert (ov.to_numpy() == ov.to_numpy().T).all()
    for col in X.columns:                                   # diagonal = own observed count
        assert ov.loc[col, col] == int(X[col].notna().sum())
    assert (ov["z_000"] == 0).all()                         # all-missing column overlaps with nothing
    assert ov.to_numpy().dtype == np.int64


def test_top_pairs_unchanged_without_overlap_and_extended_with_it():
    X = _overlap_frame()
    rho = rel.pairwise_rank_correlation(X)
    legacy = rel.top_correlated_pairs(rho, k=3)
    assert all(set(p) == {"a", "b", "rho"} for p in legacy)         # exact legacy shape

    ov = rel.pairwise_overlap_counts(X)
    extended = rel.top_correlated_pairs(rho, k=3, overlap=ov)
    assert [(p["a"], p["b"], p["rho"]) for p in extended] == [(p["a"], p["b"], p["rho"]) for p in legacy]
    for p in extended:
        assert p["n_overlap"] == int((X[p["a"]].notna() & X[p["b"]].notna()).sum())
    assert min(p["n_overlap"] for p in extended) < max(p["n_overlap"] for p in extended)


def test_thin_support_pair_reports_its_small_overlap():
    rng = np.random.default_rng(12)
    n = 2000
    t = rng.normal(size=n)
    a = np.full(n, np.nan)
    a[:150] = t[:150]                                        # a observed in rows 0..149
    b = np.full(n, np.nan)
    b[:150] = t[:150] + rng.normal(0, 0.01, 150)             # b agrees with a on the shared rows...
    b[150:300] = rng.normal(size=150)                        # ...and has extra rows outside the overlap
    X = pd.DataFrame({"a_000": a, "b_000": b, "c_000": rng.normal(size=n)})

    ov = rel.pairwise_overlap_counts(X)
    assert ov.loc["a_000", "b_000"] == 150
    top = rel.top_correlated_pairs(rel.pairwise_rank_correlation(X), k=1, overlap=ov)[0]
    assert {top["a"], top["b"]} == {"a_000", "b_000"} and top["n_overlap"] == 150 and abs(top["rho"]) > 0.99
    # below the minimum overlap the pair is not scored at all
    assert np.isnan(rel.pairwise_rank_correlation(X, min_overlap=200).loc["a_000", "b_000"])


def test_cluster_min_overlap_reflects_the_thinnest_pair():
    X = _overlap_frame()
    rho = rel.pairwise_rank_correlation(X)
    legacy = rel.near_duplicate_clusters(rho)
    assert all(set(c) == {"columns", "size", "min_pairwise_abs_rho"} for c in legacy)   # legacy shape

    ov = rel.pairwise_overlap_counts(X)
    extended = rel.near_duplicate_clusters(rho, overlap=ov)
    assert [(c["columns"], c["size"], c["min_pairwise_abs_rho"]) for c in extended] == \
           [(c["columns"], c["size"], c["min_pairwise_abs_rho"]) for c in legacy]
    cluster = next(c for c in extended if "a_000" in c["columns"])
    members = cluster["columns"]
    manual = min(
        int((X[p].notna() & X[q].notna()).sum())
        for k, p in enumerate(members) for q in members[k + 1:]
    )
    assert cluster["min_pairwise_overlap"] == manual
    assert manual < int(X["b_000"].notna().sum())            # the thin column (a) limits the overlap


def test_report_carries_new_fields_and_keeps_old_ones():
    X = _overlap_frame()
    report = rel.relationships_report(X)
    assert all("n_overlap" in p for p in report["top_correlated_pairs"])
    assert all("min_pairwise_overlap" in c for c in report["near_duplicate_clusters"])
    assert all(isinstance(p["n_overlap"], int) for p in report["top_correlated_pairs"])
    # every pre-existing key is still present with unchanged meaning
    for key in ("policy", "method", "min_overlap_rows", "near_duplicate_min_abs_rho", "n_pairs_scored",
                "n_pairs_abs_rho_ge_0_9", "n_pairs_abs_rho_ge_0_95", "near_duplicate_clusters",
                "top_correlated_pairs", "histogram_groups"):
        assert key in report
    legacy_pairs = rel.top_correlated_pairs(rel.pairwise_rank_correlation(X))
    assert [(p["a"], p["b"], p["rho"]) for p in report["top_correlated_pairs"]] == \
           [(p["a"], p["b"], p["rho"]) for p in legacy_pairs]
