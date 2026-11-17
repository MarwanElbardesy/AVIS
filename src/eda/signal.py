"""Univariate signal per feature (TRAINING data only, DESCRIPTIVE ONLY).

For each feature: the AUC of its observed values against the failure label,
its uncertainty, how stable it is across stratified folds, and how much
support (observed failures) it has.

IMPORTANT - how this must (not) be used
* These rankings are computed on the whole training set. Using them to pick
  features for a model would leak validation information into training.
  Any feature selection must be re-derived inside cross-validation folds.
* Each AUC is computed on the rows where the feature is observed, so AUCs of
  columns with different observed fractions are not directly comparable;
  `n_pos_obs` and `low_support` are reported for that reason.
* The missing-indicator AUC (2b) is shown next to the value AUC because for
  this dataset *whether* a value exists is often as informative as its value.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

from src.eda.schema import group_features

LOW_SUPPORT_POSITIVES = 30
AUC_BANDS = [(0.9, "ge_0.90"), (0.8, "0.80_to_0.90"), (0.7, "0.70_to_0.80"), (0.6, "0.60_to_0.70"), (0.0, "lt_0.60")]
TOP_K = 20


def mann_whitney_auc(values, labels):
    """AUC of `values` as a score for label==1 (ties count 1/2).

    Equivalent to sklearn.metrics.roc_auc_score; computed from average ranks.
    Returns (auc, n_pos, n_neg); auc is NaN if a class is absent.
    """

    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    ranks = rankdata(values, method="average")
    auc = (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), n_pos, n_neg


def hanley_mcneil_se(auc, n_pos, n_neg):
    """Hanley & McNeil (1982) standard error of an AUC."""

    if np.isnan(auc) or n_pos == 0 or n_neg == 0:
        return float("nan")
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def _band(oriented_auc):
    for lower, label in AUC_BANDS:
        if oriented_auc >= lower:
            return label
    return AUC_BANDS[-1][1]


def signal_report(X, y, seed=42, n_splits=5):
    """Per-feature univariate signal table plus summary (descriptive only)."""

    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("y must contain both classes encoded as 0/1.")

    splits = min(n_splits, int(y.sum()))
    folds = list(StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))

    pos = y == 1
    n_pos_total, n_neg_total = int(pos.sum()), int((~pos).sum())
    rows = {}

    for column in X.columns:
        v = X[column].to_numpy(dtype=float)
        observed = ~np.isnan(v)
        row = {"n_observed": int(observed.sum()), "n_pos_obs": int((observed & pos).sum())}
        row["n_neg_obs"] = row["n_observed"] - row["n_pos_obs"]

        # indicator AUC: P(missing | failure) vs P(missing | non-failure)
        miss_pos = 1 - row["n_pos_obs"] / n_pos_total
        miss_neg = 1 - row["n_neg_obs"] / n_neg_total
        row["indicator_auc"] = 0.5 + (miss_pos - miss_neg) / 2.0

        usable = observed.sum() > 1 and np.unique(v[observed]).size > 1
        if usable:
            auc, n1, n0 = mann_whitney_auc(v[observed], pos[observed])
        else:
            auc, n1, n0 = float("nan"), row["n_pos_obs"], row["n_neg_obs"]

        row["auc"] = auc
        row["auc_se"] = hanley_mcneil_se(auc, n1, n0)

        fold_aucs = []
        if usable:
            for _, val in folds:
                obs_val = observed[val]
                a, _, _ = mann_whitney_auc(v[val][obs_val], pos[val][obs_val])
                fold_aucs.append(a)
        fold_aucs = np.array(fold_aucs, dtype=float)
        finite = fold_aucs[~np.isnan(fold_aucs)]
        row["auc_fold_min"] = float(finite.min()) if len(finite) else float("nan")
        row["auc_fold_max"] = float(finite.max()) if len(finite) else float("nan")
        row["fold_sign_consistent"] = bool(len(finite) > 0 and ((finite > 0.5).all() or (finite < 0.5).all()))
        rows[column] = row

    table = pd.DataFrame.from_dict(rows, orient="index")
    table["auc_oriented"] = np.maximum(table["auc"], 1 - table["auc"])
    table["direction"] = np.where(
        table["auc"].isna(), "none", np.where(table["auc"] >= 0.5, "higher_in_failures", "lower_in_failures")
    )
    table["indicator_auc_oriented"] = np.maximum(table["indicator_auc"], 1 - table["indicator_auc"])
    table["low_support"] = table["n_pos_obs"] < LOW_SUPPORT_POSITIVES

    ranked = table.dropna(subset=["auc_oriented"]).sort_values(
        ["auc_oriented"], ascending=False, kind="mergesort"
    )
    bands = {label: 0 for _, label in AUC_BANDS}
    for value in ranked["auc_oriented"]:
        bands[_band(value)] += 1

    groups, _ = group_features(list(X.columns))
    by_group = {}
    for prefix, members in groups.items():
        sub = table.loc[members, "auc_oriented"].dropna()
        if len(sub):
            by_group[prefix] = {
                "n_features": len(members),
                "best_auc_oriented": float(sub.max()),
                "best_column": str(sub.idxmax()),
                "n_ge_0_7": int((sub >= 0.7).sum()),
            }

    top = ranked.head(TOP_K)
    return {
        "policy": (
            "Descriptive only. Computed on the full training set, so using these rankings to select "
            "features would leak; any selection must be re-derived inside CV folds."
        ),
        "n_folds": splits,
        "low_support_positives": LOW_SUPPORT_POSITIVES,
        "n_columns_scored": int(len(ranked)),
        "n_columns_not_scorable": int(len(table) - len(ranked)),
        "n_low_support": int(table["low_support"].sum()),
        "auc_band_counts": bands,
        "top_columns": [
            {
                "column": str(c),
                "auc_oriented": float(r["auc_oriented"]),
                "direction": r["direction"],
                "n_pos_obs": int(r["n_pos_obs"]),
                "low_support": bool(r["low_support"]),
                "fold_sign_consistent": bool(r["fold_sign_consistent"]),
            }
            for c, r in top.iterrows()
        ],
        "by_group": by_group,
        "columns": table,
    }
