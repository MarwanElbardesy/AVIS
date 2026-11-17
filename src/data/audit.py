"""Data-quality audit helpers. Run these on the TRAINING set only."""

import numpy as np
import pandas as pd


def constant_columns(X):
    """Columns with at most one distinct non-null value (e.g. cd_000)."""

    return [col for col in X.columns if X[col].nunique(dropna=True) <= 1]


def missingness_report(X, y):
    """Per-feature missing rate overall and within each class.

    `missing_lift` = missing rate among positives / missing rate among
    negatives. Values far from 1 mean missingness itself carries signal,
    which is why the pipeline keeps missing-value indicators.
    """

    y = pd.Series(np.asarray(y), index=X.index)
    is_pos = y == 1

    overall = X.isna().mean()
    pos_rate = X[is_pos].isna().mean()
    neg_rate = X[~is_pos].isna().mean()

    report = pd.DataFrame(
        {
            "missing_rate": overall,
            "missing_rate_pos": pos_rate,
            "missing_rate_neg": neg_rate,
        }
    )
    report["missing_lift"] = report["missing_rate_pos"] / report[
        "missing_rate_neg"
    ].replace(0, np.nan)

    return report.sort_values("missing_rate", ascending=False)


def audit_training_data(X, y):
    """Compact audit summary (training data only)."""

    report = missingness_report(X, y)

    return {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "positives": int(np.sum(y)),
        "positive_rate": float(np.mean(y)),
        "features_with_missing": int((X.isna().sum() > 0).sum()),
        "constant_columns": constant_columns(X),
        "duplicate_rows": int(X.duplicated().sum()),
        "top_missingness_lift": (
            report["missing_lift"].dropna().sort_values(ascending=False).head(10).round(3).to_dict()
        ),
    }
