"""Data-quality checks. Every function takes TRAINING data only.

These checks are descriptive: they flag candidates for later decisions but
never modify the data and never drop a column. In particular, sparse columns
are *not* flagged for removal - missingness is informative for this dataset.
"""

import math

import numpy as np
import pandas as pd

from src.data.audit import constant_columns  # reused from Milestone 1, unchanged

NEAR_CONSTANT_TOP_SHARE = 0.999
SATURATION_MIN_SHARE = 0.001
SATURATION_MIN_UNIQUE = 20


def non_numeric_columns(X):
    """Columns that did not parse as numbers (stray strings would show up here)."""

    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def all_missing_columns(X):
    return [c for c in X.columns if X[c].isna().all()]


def top_value_share(series):
    """Share of the most frequent non-null value among non-null values."""

    counts = series.value_counts(dropna=True, normalize=True)
    return float(counts.iloc[0]) if len(counts) else float("nan")


def near_constant_columns(X, min_top_share=NEAR_CONSTANT_TOP_SHARE):
    """Columns (including constant ones) where one value covers >= min_top_share
    of the non-null rows. Returns {column: top_value_share}."""

    result = {}
    for column in X.columns:
        share = top_value_share(X[column])
        if not math.isnan(share) and share >= min_top_share:
            result[column] = share
    return result


def duplicate_columns(X):
    """Groups of columns holding identical values (NaN positions included).

    Hash first for speed, then confirm with an exact comparison so a hash
    collision can never produce a false report.
    """

    buckets = {}
    for column in X.columns:
        hashed = pd.util.hash_pandas_object(X[column], index=False)
        key = (int(hashed.to_numpy().sum(dtype=np.uint64)), int(X[column].notna().sum()))
        buckets.setdefault(key, []).append(column)

    groups = []
    for members in buckets.values():
        remaining = list(members)
        while remaining:
            head = remaining.pop(0)
            same = [c for c in remaining if X[head].equals(X[c])]
            if same:
                groups.append([head] + same)
                remaining = [c for c in remaining if c not in same]
    return sorted(groups)


def value_profile(X):
    """One row per column: basic facts used by later checkpoints and the report."""

    rows = {}
    n_rows = len(X)

    for column in X.columns:
        s = X[column]
        non_null = s.dropna()
        numeric = pd.api.types.is_numeric_dtype(s)
        row = {
            "n_missing": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "n_unique": int(non_null.nunique()),
            "top_value_share": top_value_share(s),
        }
        if numeric and len(non_null):
            values = non_null.to_numpy(dtype=float)
            maximum = float(values.max())
            row.update(
                {
                    "min": float(values.min()),
                    "max": maximum,
                    "n_negative": int((values < 0).sum()),
                    "zero_fraction": float((values == 0).sum() / n_rows),
                    "integer_valued": bool(np.all(np.mod(values, 1) == 0)),
                    "share_at_max": float((values == maximum).sum() / n_rows),
                }
            )
        rows[column] = row

    return pd.DataFrame.from_dict(rows, orient="index")


def saturation_candidates(profile):
    """Columns with a spike at their maximum value (possible clipping or
    sentinel). Low-cardinality columns are excluded because a repeated max is
    normal there."""

    mask = (
        (profile["share_at_max"] >= SATURATION_MIN_SHARE)
        & (profile["n_unique"] >= SATURATION_MIN_UNIQUE)
    )
    return {c: float(profile.loc[c, "share_at_max"]) for c in profile.index[mask.fillna(False)]}


def wilson_interval(successes, n, z=1.96):
    """Wilson score interval for a proportion (well behaved for rare classes)."""

    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def class_balance(y):
    """Class counts, positive rate with a Wilson 95% CI, imbalance ratio."""

    y = np.asarray(y).astype(int)
    n = int(len(y))
    positives = int(y.sum())
    negatives = n - positives
    low, high = wilson_interval(positives, n)

    return {
        "n": n,
        "negatives": negatives,
        "positives": positives,
        "positive_rate": positives / n if n else float("nan"),
        "positive_rate_ci95": [low, high],
        "imbalance_ratio": negatives / positives if positives else float("nan"),
    }


def quality_report(X, y):
    """Aggregate every quality check into one JSON-friendly dict."""

    profile = value_profile(X)
    near_constant = near_constant_columns(X)
    constants = constant_columns(X)

    return {
        "class_balance": class_balance(y),
        "shape": {"rows": int(X.shape[0]), "features": int(X.shape[1])},
        "missing_cells": int(X.isna().sum().sum()),
        "features_with_missing": int((X.isna().sum() > 0).sum()),
        "duplicate_rows": int(X.duplicated().sum()),
        "constant_columns": sorted(constants),
        "near_constant_columns": dict(sorted(near_constant.items())),
        "all_missing_columns": sorted(all_missing_columns(X)),
        "non_numeric_columns": sorted(non_numeric_columns(X)),
        "duplicate_column_groups": duplicate_columns(X),
        "columns_with_negative_values": sorted(
            profile.index[profile["n_negative"].fillna(0) > 0]
        ),
        "non_integer_columns": sorted(
            profile.index[profile["integer_valued"].fillna(True).eq(False)]
        ),
        "saturation_candidates": dict(sorted(saturation_candidates(profile).items())),
        "column_profile": profile,
    }
