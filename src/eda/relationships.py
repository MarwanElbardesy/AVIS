"""Feature-feature relationships (TRAINING data only, descriptive).

* A fast, deterministic rank-correlation matrix that tolerates missing values.
* Clusters of near-duplicate (highly rank-correlated) features.
* Verification of the histogram-group candidates found by schema.py.

Nothing here removes a feature. Redundant features are reported, not dropped.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src.eda.schema import describe_groups

MIN_OVERLAP = 100          # pairs observed together in fewer rows are not scored
NEAR_DUPLICATE_RHO = 0.95  # |rho| average-linkage cut for near-duplicate clusters
TOP_PAIRS = 20


def pairwise_rank_correlation(X, min_overlap=MIN_OVERLAP):
    """Spearman-style correlation matrix for data with missing values.

    Each column is ranked once over its observed values (average ranks for
    ties), then Pearson correlation is computed over rows where BOTH columns
    are observed, using matrix products (no Python loop over pairs).

    * With no missing values this is exactly Spearman's rho.
    * With missing values it is a close approximation to pairwise Spearman
      (ranks come from each column's full observed set, not the pair's overlap).
    Pairs with fewer than `min_overlap` shared rows, and zero-variance columns,
    give NaN.
    """

    ranks = X.rank(method="average")
    R = ranks.to_numpy(dtype=float)
    observed = ~np.isnan(R)

    counts = observed.sum(axis=0)
    sums = np.where(observed, R, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros(len(counts)), where=counts > 0)  # all-missing column -> 0
    Z = np.where(observed, R - means, 0.0)         # centred per column: avoids cancellation
    O = observed.astype(float)

    n = O.T @ O
    sx = Z.T @ O                       # sum of x_i over rows where j is also observed
    sxx = (Z**2).T @ O
    sxy = Z.T @ Z

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy / n - (sx * sx.T) / n**2
        var = sxx / n - (sx / n) ** 2
        rho = cov / np.sqrt(var * var.T)

    rho[n < min_overlap] = np.nan
    np.fill_diagonal(rho, 1.0)
    rho = np.clip(rho, -1.0, 1.0)
    out = pd.DataFrame(rho, index=X.columns, columns=X.columns)

    # a constant column has no rank correlation with anything, itself included
    constant = X.nunique(dropna=True).le(1).to_numpy()
    out.loc[constant, :] = np.nan
    out.loc[:, constant] = np.nan
    return out


def pairwise_overlap_counts(X):
    """Number of rows where BOTH columns of each pair are observed (the sample
    behind each correlation in `pairwise_rank_correlation`). Integer DataFrame;
    the diagonal is each column's own observed count."""

    observed = X.notna().to_numpy(dtype=float)
    counts = observed.T @ observed
    return pd.DataFrame(counts.astype(np.int64), index=X.columns, columns=X.columns)


def near_duplicate_clusters(rho, min_abs_rho=NEAR_DUPLICATE_RHO, overlap=None):
    """Average-linkage clusters of features with |rho| >= min_abs_rho.

    Returns clusters of >= 2 features, largest first (deterministic).
    If `overlap` (from `pairwise_overlap_counts`) is given, each cluster also
    reports `min_pairwise_overlap`: the fewest shared observed rows behind any
    pair inside it. Without it the output is unchanged.
    """

    usable = [c for c in rho.columns if rho[c].notna().any() and not np.isnan(rho.loc[c, c])]
    if len(usable) < 2:
        return []

    sub = rho.loc[usable, usable].to_numpy()
    dist = np.where(np.isnan(sub), 1.0, 1.0 - np.abs(sub))
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)

    labels = fcluster(
        linkage(squareform(dist, checks=False), method="average"),
        t=1.0 - min_abs_rho,
        criterion="distance",
    )

    clusters = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 2:
            continue
        members = [usable[i] for i in idx]
        block = np.abs(sub[np.ix_(idx, idx)])
        off = block[~np.eye(len(idx), dtype=bool)]
        entry = {
            "columns": members,
            "size": len(members),
            "min_pairwise_abs_rho": float(np.nanmin(off)) if np.isfinite(off).any() else float("nan"),
        }
        if overlap is not None:
            shared = overlap.loc[members, members].to_numpy()
            entry["min_pairwise_overlap"] = int(shared[~np.eye(len(idx), dtype=bool)].min())
        clusters.append(entry)
    clusters.sort(key=lambda c: (-c["size"], c["columns"][0]))
    return clusters


def top_correlated_pairs(rho, k=TOP_PAIRS, overlap=None):
    """The k pairs with the largest |rho| (ties broken by name).

    If `overlap` is given, each pair also reports `n_overlap`, the number of
    rows where both features are observed. Without it the output is unchanged.
    """

    cols = list(rho.columns)
    values = rho.to_numpy()
    i, j = np.triu_indices(len(cols), k=1)
    strength = np.abs(values[i, j])
    keep = np.flatnonzero(~np.isnan(strength))
    order = sorted(keep, key=lambda t: (-strength[t], cols[i[t]], cols[j[t]]))
    pairs = []
    for t in order[:k]:
        entry = {"a": cols[i[t]], "b": cols[j[t]], "rho": float(values[i[t], j[t]])}
        if overlap is not None:
            entry["n_overlap"] = int(overlap.loc[cols[i[t]], cols[j[t]]])
        pairs.append(entry)
    return pairs


def verify_histogram_groups(X, rho, bins=None):
    """Check structural histogram candidates against the actual values.

    Per candidate group: on rows where every bin is observed, how consistent is
    the row total (a true count histogram over a fixed population would be
    constant), how often is the whole histogram zero, are any bins negative,
    and how strongly do neighbouring bins correlate?
    """

    info = describe_groups(list(X.columns)) if bins is None else describe_groups(list(X.columns), bins=bins)
    result = {}

    for prefix in info["histogram_candidate_prefixes"]:
        cols = info["groups"][prefix]["columns"]
        block = X[cols]
        complete = block.notna().all(axis=1)
        totals = block[complete].sum(axis=1)

        entry = {
            "columns": cols,
            "n_complete_rows": int(complete.sum()),
            "complete_share": float(complete.mean()),
            "any_negative_values": bool((block < 0).any().any()),
        }

        if len(totals):
            mean = float(totals.mean())
            modal = totals.value_counts().iloc[0]
            entry.update(
                {
                    "row_total_min": float(totals.min()),
                    "row_total_median": float(totals.median()),
                    "row_total_max": float(totals.max()),
                    "row_total_cv": float(totals.std(ddof=0) / mean) if mean > 0 else float("nan"),
                    "modal_total_share": float(modal / len(totals)),
                    "total_is_constant": bool(totals.nunique() == 1),
                    "all_zero_share": float((totals == 0).mean()),
                }
            )

        adjacent = [rho.loc[cols[k], cols[k + 1]] for k in range(len(cols) - 1)]
        adjacent = [a for a in adjacent if not np.isnan(a)]
        entry["mean_adjacent_bin_rho"] = float(np.mean(adjacent)) if adjacent else float("nan")
        result[prefix] = entry

    return result


def relationships_report(X):
    """Full relationships section (no labels involved)."""

    rho = pairwise_rank_correlation(X)
    overlap = pairwise_overlap_counts(X)
    clusters = near_duplicate_clusters(rho, overlap=overlap)
    values = rho.to_numpy()
    upper = np.abs(values[np.triu_indices(len(rho), k=1)])
    upper = upper[~np.isnan(upper)]

    return {
        "policy": "Descriptive only. Redundant features are reported, never dropped.",
        "method": "per-column average ranks, Pearson over rows where both are observed "
                  "(exact Spearman when nothing is missing)",
        "min_overlap_rows": MIN_OVERLAP,
        "near_duplicate_min_abs_rho": NEAR_DUPLICATE_RHO,
        "n_pairs_scored": int(len(upper)),
        "n_pairs_abs_rho_ge_0_9": int((upper >= 0.9).sum()),
        "n_pairs_abs_rho_ge_0_95": int((upper >= 0.95).sum()),
        "near_duplicate_clusters": clusters,
        "top_correlated_pairs": top_correlated_pairs(rho, overlap=overlap),
        "histogram_groups": verify_histogram_groups(X, rho),
    }
