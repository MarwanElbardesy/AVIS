"""Leakage and ordering risk checks (TRAINING data only).

Looks for ways a model (or a cross-validation split) could be fooled:

* row order related to the label (a sorted or time-ordered file),
* features that drift with row position,
* features that (almost) separate the classes by themselves,
* values that occur in only one class,
* duplicated feature rows, especially with conflicting labels.

Limitation, stated plainly: the APS data has no vehicle identifier, so
near-duplicate snapshots of the same truck cannot be detected or grouped.
Plain stratified CV can therefore still be somewhat optimistic; the only
real remedy is the final evaluation on the untouched official test set.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm, spearmanr

N_BLOCKS = 10
BLOCK_P = 1e-3
BLOCK_CRAMERS_V = 0.03
RUNS_P = 1e-3
DRIFT_ABS_RHO = 0.05
SUSPICIOUS_VALUE_AUC = 0.98
SUSPICIOUS_INDICATOR_AUC = 0.90
EXCLUSIVE_MAX_UNIQUE = 20
EXCLUSIVE_MIN_COUNT = 20


def block_positive_rates(y, n_blocks=N_BLOCKS):
    """Positive rate in consecutive equal blocks of the file, with a
    chi-square test of homogeneity and Cramer's V (effect size)."""

    y = np.asarray(y).astype(int)
    blocks = np.array_split(np.arange(len(y)), n_blocks)
    pos = np.array([int(y[b].sum()) for b in blocks])
    size = np.array([len(b) for b in blocks])
    table = np.vstack([pos, size - pos])

    chi2, p, _, _ = chi2_contingency(table)
    v = float(np.sqrt(chi2 / table.sum()))   # min(rows-1, cols-1) == 1 for a 2-row table
    return {
        "n_blocks": n_blocks,
        "positive_rate_by_block": (pos / size).tolist(),
        "chi2": float(chi2),
        "p_value": float(p),
        "cramers_v": v,
    }


def runs_test(y):
    """Wald-Wolfowitz runs test: are labels in random order along the file?

    Too few runs => labels are clustered (e.g. sorted); too many => alternating.
    """

    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    n = n1 + n0
    runs = 1 + int((y[1:] != y[:-1]).sum())
    expected = 2 * n1 * n0 / n + 1
    variance = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n**2 * (n - 1))
    z = (runs - expected) / np.sqrt(variance) if variance > 0 else float("nan")
    return {
        "runs": runs,
        "expected_runs": float(expected),
        "z": float(z),
        "p_value": float(2 * norm.sf(abs(z))) if not np.isnan(z) else float("nan"),
    }


def feature_index_drift(X, top_k=5):
    """Spearman correlation between each feature (observed rows) and row position."""

    position = np.arange(len(X), dtype=float)
    rhos = {}
    for column in X.columns:
        v = X[column].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        if ok.sum() > 2 and np.unique(v[ok]).size > 1:
            rhos[column] = float(spearmanr(position[ok], v[ok]).statistic)

    series = pd.Series(rhos, dtype=float)
    order = series.abs().sort_values(ascending=False, kind="mergesort")

    counts = X.isna().sum(axis=1).to_numpy(dtype=float)
    missing_rho = float(spearmanr(position, counts).statistic) if np.unique(counts).size > 1 else float("nan")

    return {
        "n_features_tested": int(len(series)),
        "max_abs_rho": float(order.iloc[0]) if len(order) else float("nan"),
        "n_abs_rho_ge_threshold": int((series.abs() >= DRIFT_ABS_RHO).sum()),
        "top_drifting_columns": [{"column": c, "rho": float(series[c])} for c in order.head(top_k).index],
        "row_missing_count_vs_position_rho": missing_rho,
    }


def class_exclusive_values(X, y, max_unique=EXCLUSIVE_MAX_UNIQUE, min_count=EXCLUSIVE_MIN_COUNT):
    """Values of low-cardinality columns seen in only one class (with enough rows)."""

    y = np.asarray(y).astype(int)
    found = {}
    for column in X.columns:
        s = X[column]
        n_unique = s.nunique(dropna=True)
        if not 2 <= n_unique <= max_unique:
            continue
        counts = pd.crosstab(s.dropna(), y[s.notna().to_numpy()])
        for col in (0, 1):
            if col not in counts.columns:
                counts[col] = 0
        for value, row in counts.iterrows():
            total = int(row[0] + row[1])
            if total >= min_count and (row[0] == 0 or row[1] == 0):
                found.setdefault(column, []).append(
                    {"value": float(value), "only_class": int(1 if row[0] == 0 else 0), "count": total}
                )
    return {c: found[c][:10] for c in sorted(found)}


def duplicate_row_check(X, y):
    """Duplicated feature rows, and how many of them carry conflicting labels."""

    y = np.asarray(y).astype(int)
    hashes = pd.util.hash_pandas_object(X, index=False).to_numpy()
    frame = pd.DataFrame({"h": hashes, "y": y})
    dup_mask = frame.duplicated("h", keep=False)
    conflicts = frame[dup_mask].groupby("h")["y"].nunique()
    return {
        "n_rows_in_duplicate_groups": int(dup_mask.sum()),
        "n_conflicting_label_groups": int((conflicts > 1).sum()),
    }


def suspicious_single_features(signal_table):
    """Features whose value-AUC or missing-indicator AUC is implausibly high."""

    value = signal_table[signal_table["auc_oriented"] >= SUSPICIOUS_VALUE_AUC]
    indicator = signal_table[signal_table["indicator_auc_oriented"] >= SUSPICIOUS_INDICATOR_AUC]
    return {
        "value_auc_threshold": SUSPICIOUS_VALUE_AUC,
        "indicator_auc_threshold": SUSPICIOUS_INDICATOR_AUC,
        "value_auc": sorted(value.index.tolist()),
        "indicator_auc": sorted(indicator.index.tolist()),
    }


def leakage_report(X, y, signal_table):
    """All leakage / ordering checks. `signal_table` comes from signal.signal_report."""

    blocks = block_positive_rates(y)
    runs = runs_test(y)
    drift = feature_index_drift(X)

    flags = {
        "row_order_related_to_label": bool(
            (blocks["p_value"] < BLOCK_P and blocks["cramers_v"] >= BLOCK_CRAMERS_V)
            or (not np.isnan(runs["p_value"]) and runs["p_value"] < RUNS_P)
        ),
        "features_drift_with_row_position": drift["n_abs_rho_ge_threshold"] > 0,
    }
    suspicious = suspicious_single_features(signal_table)
    flags["suspiciously_strong_single_feature"] = bool(suspicious["value_auc"] or suspicious["indicator_auc"])
    exclusive = class_exclusive_values(X, y)
    flags["class_exclusive_values_present"] = bool(exclusive)
    dups = duplicate_row_check(X, y)
    flags["duplicate_rows_with_conflicting_labels"] = dups["n_conflicting_label_groups"] > 0

    return {
        "policy": (
            "Flags are review prompts, not verdicts. A flag means 'look closer', and the absence of a flag "
            "does not rule out same-vehicle leakage, which APS data cannot reveal (no vehicle id)."
        ),
        "thresholds": {
            "block_p": BLOCK_P,
            "block_cramers_v": BLOCK_CRAMERS_V,
            "runs_p": RUNS_P,
            "drift_abs_rho": DRIFT_ABS_RHO,
        },
        "flags": flags,
        "row_order": {"blocks": blocks, "runs_test": runs},
        "feature_drift": drift,
        "suspicious_single_features": suspicious,
        "class_exclusive_values": exclusive,
        "duplicate_rows": dups,
    }
