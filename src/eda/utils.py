"""Small shared helpers for the EDA package (hashing, JSON-safe conversion)."""

import hashlib
import json
import math
import platform

import numpy as np
import pandas as pd


def sha256_file(path, chunk_size=1 << 20):
    """SHA-256 of a file, used to tie every EDA result to the exact input data."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonable(obj):
    """Recursively convert numpy/pandas objects to plain JSON types.

    NaN and +/-inf become None so the output is strict, deterministic JSON.
    """

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        items = sorted(obj) if isinstance(obj, set) else obj
        return [to_jsonable(v) for v in items]
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="index"))
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def dumps_deterministic(obj):
    """Stable JSON text: sorted keys, fixed indent, trailing newline."""

    return json.dumps(to_jsonable(obj), indent=2, sort_keys=True, allow_nan=False) + "\n"


def library_versions():
    """Versions that can influence numeric results (recorded for reproducibility)."""

    import sklearn

    versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except ImportError:  # scipy is only required from checkpoint 2b onward
        pass
    return versions
