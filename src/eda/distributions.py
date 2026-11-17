"""Per-feature distribution profile (TRAINING data only, descriptive).

APS features are anonymized counters and histogram bins, typically heavy
tailed and zero inflated. This module measures that, so later preprocessing
choices (scaling, transforms, tree vs linear models) rest on evidence. It
never transforms or drops anything.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

HEAVY_SKEW = 5.0          # |skew| at or above this => "heavy tailed"
ZERO_INFLATED = 0.5       # share of observed values equal to 0
QUANTILES = {"q01": 0.01, "q05": 0.05, "q25": 0.25, "median": 0.5, "q75": 0.75, "q95": 0.95, "q99": 0.99}


def column_distribution(values):
    """Distribution facts for one array of observed (non-null) values."""

    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"n_observed": 0}

    q = {name: float(np.quantile(values, p)) for name, p in QUANTILES.items()}
    iqr = q["q75"] - q["q25"]
    std = float(values.std(ddof=1)) if n > 1 else float("nan")
    maximum = float(values.max())

    row = {
        "n_observed": int(n),
        "mean": float(values.mean()),
        "std": std,
        "min": float(values.min()),
        **q,
        "max": maximum,
        "iqr": float(iqr),
        "zero_fraction_observed": float((values == 0).mean()),
        "max_over_q99": float(maximum / q["q99"]) if q["q99"] > 0 else float("nan"),
    }

    constant = std == 0 or n < 4
    row["skew"] = float("nan") if constant else float(skew(values, bias=False))
    row["excess_kurtosis"] = float("nan") if constant else float(kurtosis(values, bias=False))

    # With a zero IQR the Tukey fences collapse onto the median and "outliers"
    # would just mean "not the median", which is misleading; report NaN instead.
    if iqr > 0:
        for name, k in (("outlier_share_1_5iqr", 1.5), ("outlier_share_3iqr", 3.0)):
            lo, hi = q["q25"] - k * iqr, q["q75"] + k * iqr
            row[name] = float(((values < lo) | (values > hi)).mean())
    else:
        row["outlier_share_1_5iqr"] = float("nan")
        row["outlier_share_3iqr"] = float("nan")
    row["iqr_is_zero"] = bool(iqr == 0)
    return row


def distribution_report(X, y):
    """Distribution table for every column plus flag counts.

    Class-conditional medians are descriptive only (they are not a selection
    criterion; see signal.py for the caveats on univariate rankings).
    """

    y = np.asarray(y).astype(int)
    pos = y == 1

    rows = {}
    for column in X.columns:
        s = X[column].to_numpy(dtype=float)
        observed = ~np.isnan(s)
        row = column_distribution(s[observed])
        if row["n_observed"]:
            pos_vals, neg_vals = s[observed & pos], s[observed & ~pos]
            row["median_pos"] = float(np.median(pos_vals)) if len(pos_vals) else float("nan")
            row["median_neg"] = float(np.median(neg_vals)) if len(neg_vals) else float("nan")
        rows[column] = row

    table = pd.DataFrame.from_dict(rows, orient="index")
    has = table["n_observed"] > 0

    table["heavy_tailed"] = (table["skew"].abs() >= HEAVY_SKEW).fillna(False)
    table["zero_inflated"] = (table["zero_fraction_observed"] >= ZERO_INFLATED).fillna(False)

    ranked = table["skew"].abs().dropna().sort_values(ascending=False, kind="mergesort")

    return {
        "policy": "Descriptive only. No transformation, scaling or removal is applied or implied.",
        "thresholds": {"heavy_skew": HEAVY_SKEW, "zero_inflated": ZERO_INFLATED},
        "n_columns_profiled": int(has.sum()),
        "n_heavy_tailed": int(table["heavy_tailed"].sum()),
        "n_zero_inflated": int(table["zero_inflated"].sum()),
        "n_iqr_zero": int(table["iqr_is_zero"].fillna(False).sum()),
        "most_skewed_columns": ranked.head(10).index.tolist(),
        "columns": table,
    }
