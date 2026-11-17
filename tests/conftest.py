import numpy as np
import pandas as pd
import pytest

LICENSE_HEADER = "\n".join(f"license line {i}" for i in range(20)) + "\n"


def make_aps_like(n=600, n_features=8, pos_rate=0.08, seed=0):
    """Synthetic APS-shaped data: strings 'neg'/'pos', a constant column,
    NaNs whose frequency depends on the class, and real signal in a few
    columns. Purely for tests - never used to report model performance."""

    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    y[:5] = 1  # guarantee positives

    data = {"class": np.where(y == 1, "pos", "neg")}
    for j in range(n_features):
        signal = y * (1.5 if j < 3 else 0.0)
        col = rng.normal(0, 1, n) + signal
        miss_p = 0.5 if j == 3 else 0.1
        miss_p = np.where(y == 1, min(miss_p * 3, 0.9), miss_p)
        col[rng.random(n) < miss_p] = np.nan
        data[f"f_{j}"] = col
    data["cd_000"] = 1_209_600.0
    return pd.DataFrame(data)


def write_aps_csv(df, path, header=LICENSE_HEADER):
    text = header + df.to_csv(index=False, na_rep="na")
    path.write_text(text)
    return path


@pytest.fixture
def aps_df():
    return make_aps_like()


@pytest.fixture
def xy(aps_df):
    from src.data.scania import split_features_target

    return split_features_target(aps_df)
