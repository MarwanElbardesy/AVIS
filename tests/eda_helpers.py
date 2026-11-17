"""Synthetic APS-style frame with planted structure for EDA tests.

Only used by tests. Never used to report findings about the real dataset.
"""

import numpy as np
import pandas as pd


def make_eda_frame(n=2000, seed=0, pos_rate=0.05):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    y[:10] = 1

    data = {}
    data["aa_000"] = rng.poisson(50, n).astype(float)                       # integer counter
    data["ab_000"] = np.where(rng.random(n) < 0.3, np.nan, rng.poisson(5, n)).astype(float)
    data["ac_000"] = rng.normal(0, 1, n)                                    # non-integer
    data["ad_000"] = data["aa_000"].copy()                                  # duplicate of aa_000
    data["ae_000"] = np.nan                                                 # all missing
    data["af_000"] = rng.normal(0, 1, n).round(0)                           # has negatives, integer
    data["cd_000"] = 1_209_600.0                                            # constant
    near = np.zeros(n)
    near[:1] = 3.0
    data["ca_000"] = near                                                   # near-constant (99.95% zero)
    sat = rng.integers(0, 1000, n).astype(float)
    sat[rng.random(n) < 0.02] = 1000.0                                      # spike at max, high cardinality
    data["ai_000"] = sat

    for prefix in ("ag", "ay"):                                             # 10-bin histogram groups
        for b in range(10):
            data[f"{prefix}_{b:03d}"] = rng.poisson(3 + b, n).astype(float)
    for b in range(3):                                                      # short group, not a histogram
        data[f"bb_{b:03d}"] = rng.poisson(10, n).astype(float)
    for b in (0, 2, 5, 6, 7, 8, 9, 10, 11, 12):                             # 10 columns, NOT contiguous from 0
        data[f"bc_{b:03d}"] = rng.poisson(2, n).astype(float)

    X = pd.DataFrame(data)
    return X, pd.Series(y, name="y")


def make_missingness_frame(n=6000, seed=0, pos_rate=0.08):
    """Frame with planted missingness structure (tests only).

    inf_000        missing 80% in failures, 10% in non-failures  (informative)
    nul_000        missing 20% independent of the label          (null)
    pa_000/001     identical missing mask                         (identical pattern)
    cl_000..002    one base mask, ~1% flips on two copies         (co-missing cluster, not identical)
    g_000..003     whole group missing together in 30% of rows    (structural group)
    ind_000/001    independent 15% masks                          (unrelated noise)
    full_000       never missing;  none_000 always missing
    """

    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    y[:20] = 1
    pos = y == 1

    def values():
        return rng.normal(0, 1, n)

    def masked(mask):
        v = values()
        v[mask] = np.nan
        return v

    data = {}
    data["inf_000"] = masked(rng.random(n) < np.where(pos, 0.8, 0.1))
    data["nul_000"] = masked(rng.random(n) < 0.2)

    pa_mask = rng.random(n) < 0.3
    data["pa_000"], data["pa_001"] = masked(pa_mask), masked(pa_mask)

    base = rng.random(n) < 0.25
    for k in range(3):
        mask = base.copy()
        if k > 0:
            flip = rng.random(n) < 0.01
            mask = np.where(flip, ~mask, mask)
        data[f"cl_{k:03d}"] = masked(mask)

    g_mask = rng.random(n) < 0.3
    for k in range(4):
        data[f"g_{k:03d}"] = masked(g_mask)

    data["ind_000"] = masked(rng.random(n) < 0.15)
    data["ind_001"] = masked(rng.random(n) < 0.15)
    data["full_000"] = values()
    data["none_000"] = np.nan

    return pd.DataFrame(data), pd.Series(y, name="y")
