"""Missingness analysis (TRAINING data only, purely descriptive).

Questions answered
1. How much is missing, per column and per row?
2. Is missingness related to the failure label (and how strongly)?
3. Do columns go missing together (shared sensor / subsystem outage)?
4. Is missingness structured by feature group, and which row patterns recur?
5. Are the target associations stable across stratified folds?

Policy: nothing here removes or modifies a feature. In this dataset the fact
that a value is missing carries signal, so a high missing rate is a property
to understand, not a defect to delete. Statistical significance is reported
with effect sizes because with 60,000 rows almost any association is
"significant"; none of these rankings may be used to select features unless
re-derived inside cross-validation folds.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import false_discovery_control, fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.eda.schema import describe_groups

POLICY = (
    "Descriptive only. No feature is removed or modified because of missingness; "
    "missingness is informative for this dataset. Any use of these rankings for "
    "feature selection must be re-derived inside cross-validation folds."
)

ALPHA = 0.05
NOTABLE_AUC_DELTA = 0.05      # |AUC - 0.5| of the missing-indicator considered practically notable
COMISSING_MIN_PHI = 0.9       # average-linkage cut for co-missingness clusters
TOP_PAIRS = 20
TOP_PATTERNS = 10
MAX_LISTED_COLUMNS = 15


# --------------------------------------------------------------------------- #
# Basic structure
# --------------------------------------------------------------------------- #
def _as_binary(y):
    y = np.asarray(y).astype(int)
    if not set(np.unique(y)) <= {0, 1} or y.sum() == 0 or y.sum() == len(y):
        raise ValueError("y must contain both classes encoded as 0/1.")
    return y


def indicator_matrix(X):
    """Boolean missing-indicator matrix restricted to columns whose indicator
    varies (columns that are never or always missing carry no pattern)."""

    miss = X.isna()
    varying = [c for c in X.columns if 0 < int(miss[c].sum()) < len(X)]
    return miss[varying]


def missing_rate_bucket(rate):
    if rate == 0:
        return "none"
    if rate >= 1:
        return "all"
    for upper, label in ((0.01, "(0,1%]"), (0.10, "(1,10%]"), (0.50, "(10,50%]"), (0.90, "(50,90%]")):
        if rate <= upper:
            return label
    return "(90,100%)"


def column_missing_summary(X, y):
    """Per-column counts, rate, bucket, and how many positives are observed."""

    y = _as_binary(y)
    pos = y == 1
    miss = X.isna().to_numpy()

    table = pd.DataFrame(
        {
            "n_missing": miss.sum(axis=0),
            "missing_rate": miss.mean(axis=0),
            "n_pos_missing": miss[pos].sum(axis=0),
            "n_pos_observed": (~miss[pos]).sum(axis=0),
        },
        index=X.columns,
    )
    table["bucket"] = [missing_rate_bucket(r) for r in table["missing_rate"]]
    return table


# --------------------------------------------------------------------------- #
# Association with the target
# --------------------------------------------------------------------------- #
def indicator_auc(rate_pos, rate_neg):
    """ROC-AUC of a binary missing-indicator used as a score.

    For a 0/1 score this equals 0.5 + (P(miss|pos) - P(miss|neg)) / 2.
    """

    return 0.5 + (np.asarray(rate_pos) - np.asarray(rate_neg)) / 2.0


def odds_ratio(a, b, c, d):
    """Odds ratio of [[a, b], [c, d]]; Haldane-Anscombe +0.5 if any cell is 0."""

    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return (a * d) / (b * c)


def missingness_vs_target(X, y, alpha=ALPHA):
    """Per-column association between 'value is missing' and failure.

    Columns: missing rate within positives / negatives, lift, indicator AUC,
    odds ratio, Fisher exact p-value and Benjamini-Hochberg q-value (FDR over
    all tested columns). Only columns with a varying indicator are tested.
    """

    y = _as_binary(y)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())

    M = indicator_matrix(X)
    miss = M.to_numpy()

    rows = {}
    for j, column in enumerate(M.columns):
        a = int((miss[pos, j]).sum())          # missing & positive
        b = int((miss[~pos, j]).sum())         # missing & negative
        c, d = n_pos - a, n_neg - b
        rate_pos, rate_neg = a / n_pos, b / n_neg
        rows[column] = {
            "missing_rate_pos": rate_pos,
            "missing_rate_neg": rate_neg,
            "missing_lift": rate_pos / rate_neg if rate_neg > 0 else np.nan,
            "indicator_auc": float(indicator_auc(rate_pos, rate_neg)),
            "odds_ratio": odds_ratio(a, b, c, d),
            "p_value": float(fisher_exact([[a, b], [c, d]])[1]),
        }

    table = pd.DataFrame.from_dict(rows, orient="index")
    if len(table):
        table["q_value"] = false_discovery_control(table["p_value"].to_numpy(), method="bh")
        table["significant_fdr"] = table["q_value"] < alpha
        table["notable_effect"] = (table["indicator_auc"] - 0.5).abs() >= NOTABLE_AUC_DELTA
    return table


def fold_auc_stability(X, y, n_splits=5, seed=42):
    """Indicator AUC recomputed on each stratified validation fold.

    Shows whether a missingness-target association is stable or an artefact of
    a few rows. Descriptive only: it does not select anything.
    """

    y = _as_binary(y)
    M = indicator_matrix(X)
    A = M.to_numpy()
    n_splits = min(n_splits, int(y.sum()))
    if n_splits < 2:
        raise ValueError("Need at least 2 positives for fold stability.")

    aucs = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for _, val in splitter.split(A, y):
        v, yv = A[val], y[val] == 1
        aucs.append(indicator_auc(v[yv].mean(axis=0), v[~yv].mean(axis=0)))

    aucs = np.vstack(aucs)
    return pd.DataFrame(
        {"auc_fold_min": aucs.min(axis=0), "auc_fold_max": aucs.max(axis=0)},
        index=M.columns,
    )


def row_missing_profile(X, y):
    """How many values are missing per row, overall and by class."""

    y = _as_binary(y)
    pos = y == 1
    counts = X.isna().sum(axis=1).to_numpy()
    qs = {"min": 0.0, "q25": 0.25, "median": 0.5, "q75": 0.75, "q95": 0.95, "max": 1.0}

    def describe(values):
        return {
            "mean": float(values.mean()),
            "quantiles": {name: float(np.quantile(values, q)) for name, q in qs.items()},
            "complete_row_share": float((values == 0).mean()),
        }

    return {
        "n_features": int(X.shape[1]),
        "overall": describe(counts),
        "positive": describe(counts[pos]),
        "negative": describe(counts[~pos]),
        "auc_missing_count_vs_target": float(roc_auc_score(y, counts)),
        "mannwhitney_p": float(mannwhitneyu(counts[pos], counts[~pos], alternative="two-sided").pvalue),
    }


# --------------------------------------------------------------------------- #
# Co-missingness structure
# --------------------------------------------------------------------------- #
def phi_matrix(M):
    """Phi coefficient (Pearson on 0/1) between missing indicators."""

    phi = np.corrcoef(M.to_numpy(dtype=np.float64), rowvar=False)
    return np.nan_to_num(phi, nan=0.0)


def identical_pattern_groups(M):
    """Groups of columns that are missing in exactly the same rows."""

    buckets = {}
    for column in M.columns:
        buckets.setdefault(M[column].to_numpy().tobytes(), []).append(column)
    return sorted(g for g in buckets.values() if len(g) > 1)


def comissing_clusters(M, min_phi=COMISSING_MIN_PHI):
    """Average-linkage clusters of columns whose indicators correlate >= min_phi.

    Returns clusters with >= 2 columns, largest first. Deterministic: the
    result depends only on the data and column order.
    """

    columns = list(M.columns)
    if len(columns) < 2:
        return []

    phi = phi_matrix(M)
    dist = np.clip(1.0 - phi, 0.0, None)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)

    labels = fcluster(
        linkage(squareform(dist, checks=False), method="average"),
        t=1.0 - min_phi,
        criterion="distance",
    )

    rates = M.mean()
    clusters = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 2:
            continue
        members = [columns[i] for i in idx]
        sub = phi[np.ix_(idx, idx)]
        off_diag = sub[~np.eye(len(idx), dtype=bool)]
        clusters.append(
            {
                "columns": members,
                "size": len(members),
                "mean_missing_rate": float(rates[members].mean()),
                "min_pairwise_phi": float(off_diag.min()),
            }
        )

    clusters.sort(key=lambda c: (-c["size"], c["columns"][0]))
    return clusters


def top_comissing_pairs(M, k=TOP_PAIRS):
    """The k column pairs whose missing indicators are most positively correlated."""

    columns = list(M.columns)
    if len(columns) < 2:
        return []
    phi = phi_matrix(M)
    i, j = np.triu_indices(len(columns), k=1)
    order = sorted(range(len(i)), key=lambda t: (-phi[i[t], j[t]], columns[i[t]], columns[j[t]]))
    return [
        {"a": columns[i[t]], "b": columns[j[t]], "phi": float(phi[i[t], j[t]])}
        for t in order[:k]
    ]


def group_missingness_structure(X, y):
    """Is missingness structured by anonymized feature group (prefix)?

    For each prefix group with >= 2 columns: the share of rows where the whole
    group is missing, where it is partly missing, and where it is complete;
    plus how often the whole group is missing for positives vs negatives.
    """

    y = _as_binary(y)
    pos = y == 1
    groups = describe_groups(list(X.columns))["groups"]
    result = {}

    for prefix, info in groups.items():
        cols = info["columns"]
        if len(cols) < 2:
            continue
        n_missing = X[cols].isna().sum(axis=1).to_numpy()
        all_missing = n_missing == len(cols)
        none_missing = n_missing == 0
        result[prefix] = {
            "n_features": len(cols),
            "histogram_candidate": info["histogram_candidate"],
            "all_missing_share": float(all_missing.mean()),
            "partial_missing_share": float((~all_missing & ~none_missing).mean()),
            "complete_share": float(none_missing.mean()),
            "all_missing_rate_pos": float(all_missing[pos].mean()),
            "all_missing_rate_neg": float(all_missing[~pos].mean()),
        }
    return result


def row_pattern_summary(M, y, top_k=TOP_PATTERNS):
    """Distinct row-level missingness patterns and the most frequent ones."""

    y = _as_binary(y)
    if M.shape[1] == 0:
        return {"n_distinct_patterns": 1, "top_k_coverage": 1.0, "top_patterns": []}

    packed = np.packbits(M.to_numpy(), axis=1)
    unique, inverse, counts = np.unique(packed, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)  # shape differs across numpy versions
    positives = np.bincount(inverse, weights=y, minlength=len(unique))

    n = len(y)
    order = sorted(range(len(unique)), key=lambda t: (-counts[t], unique[t].tobytes()))
    columns = np.asarray(M.columns)
    top = []

    for t in order[:top_k]:
        bits = np.unpackbits(unique[t])[: M.shape[1]].astype(bool)
        missing_columns = columns[bits].tolist()
        top.append(
            {
                "n_rows": int(counts[t]),
                "share": float(counts[t] / n),
                "positives": int(positives[t]),
                "positive_rate": float(positives[t] / counts[t]),
                "n_missing_columns": int(bits.sum()),
                "missing_columns_listed": missing_columns[:MAX_LISTED_COLUMNS],
            }
        )

    return {
        "n_distinct_patterns": int(len(unique)),
        "top_k_coverage": float(sum(counts[t] for t in order[:top_k]) / n),
        "top_patterns": top,
    }


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def _direction_splits(tested):
    """Higher/lower-missing-in-failures splits among significant and among
    notable-effect indicators (additive fields; see DECISIONS D22)."""

    keys = (
        "n_significant_higher_missing_in_positives",
        "n_significant_lower_missing_in_positives",
        "n_notable_higher_missing_in_positives",
        "n_notable_lower_missing_in_positives",
        "n_significant_and_notable",
    )
    if not len(tested):
        return {k: 0 for k in keys}

    higher = tested["indicator_auc"] > 0.5
    lower = tested["indicator_auc"] < 0.5
    significant = tested["significant_fdr"].astype(bool)
    notable = tested["notable_effect"].astype(bool)
    return {
        keys[0]: int((significant & higher).sum()),
        keys[1]: int((significant & lower).sum()),
        keys[2]: int((notable & higher).sum()),
        keys[3]: int((notable & lower).sum()),
        keys[4]: int((significant & notable).sum()),
    }


def missingness_report(X, y, seed=42, alpha=ALPHA):
    """Run every missingness analysis and return one JSON-friendly dict."""

    y = _as_binary(y)
    M = indicator_matrix(X)

    summary = column_missing_summary(X, y)
    association = missingness_vs_target(X, y, alpha=alpha)
    stability = fold_auc_stability(X, y, seed=seed)

    columns = summary.join(association).join(stability)

    tested = association
    strongest = (
        tested.assign(effect=(tested["indicator_auc"] - 0.5).abs())
        .sort_values(["effect"], ascending=False, kind="mergesort")
        .head(10)
        .index.tolist()
        if len(tested)
        else []
    )

    clusters = comissing_clusters(M)

    return {
        "policy": POLICY,
        "n_columns_total": int(X.shape[1]),
        "n_columns_with_missing": int((summary["n_missing"] > 0).sum()),
        "n_columns_all_missing": int((summary["bucket"] == "all").sum()),
        "bucket_counts": summary["bucket"].value_counts().sort_index().to_dict(),
        "target_association": {
            "alpha": alpha,
            "notable_auc_delta": NOTABLE_AUC_DELTA,
            "n_tested": int(len(tested)),
            "n_significant_fdr": int(tested["significant_fdr"].sum()) if len(tested) else 0,
            "n_notable_effect": int(tested["notable_effect"].sum()) if len(tested) else 0,
            "n_higher_missing_in_positives": int((tested["indicator_auc"] > 0.5).sum()) if len(tested) else 0,
            "n_lower_missing_in_positives": int((tested["indicator_auc"] < 0.5).sum()) if len(tested) else 0,
            **_direction_splits(tested),
            "strongest_columns": strongest,
        },
        "rows": row_missing_profile(X, y),
        "identical_pattern_groups": identical_pattern_groups(M),
        "comissing_clusters": clusters,
        "comissing_cluster_min_phi": COMISSING_MIN_PHI,
        "top_comissing_pairs": top_comissing_pairs(M),
        "group_structure": group_missingness_structure(X, y),
        "row_patterns": row_pattern_summary(M, y),
        "columns": columns,
    }
