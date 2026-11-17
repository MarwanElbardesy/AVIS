from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "scania"

TRAIN_FILE = DATA_DIR / "aps_failure_training_set.csv"
TEST_FILE = DATA_DIR / "aps_failure_test_set.csv"

TARGET = "class"
POSITIVE_LABEL = "pos"
NEGATIVE_LABEL = "neg"

# The official files ship with a 20-line license header before the CSV header.
# Used only as a fallback; the header row is normally detected automatically.
DEFAULT_HEADER_LINES = 20


def find_header_row(path, target=TARGET, max_lines=200):
    """Return the 0-based line index of the CSV header row (the line that
    starts with the target column name). Falls back to the documented
    20-line header if it cannot be found."""

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if line.startswith(f"{target},"):
                return index
            if index >= max_lines:
                break

    return DEFAULT_HEADER_LINES


def read_aps_csv(path):
    """Read one APS csv file ('na' -> NaN), skipping the license header."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path, skiprows=find_header_row(path), na_values="na")


def load_scania_train(path=None):
    """Load ONLY the training set. All model development uses this, so the
    official test set is never touched before final evaluation."""

    train = read_aps_csv(path or TRAIN_FILE)
    validate_scania_frame(train, "Training")
    return train


def load_scania_test(path=None):
    """Load the official test set. Call this only for the final evaluation."""

    test = read_aps_csv(path or TEST_FILE)
    validate_scania_frame(test, "Test")
    return test


def load_scania_data():
    """Load and validate the Scania APS training and test datasets."""

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_FILE}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_FILE}")

    train = read_aps_csv(TRAIN_FILE)
    test = read_aps_csv(TEST_FILE)

    validate_scania_data(train, test)

    return train, test


def validate_scania_frame(df, name="Dataset"):
    """Validate a single APS dataframe (schema, labels, emptiness)."""

    if TARGET not in df.columns:
        raise ValueError("Missing target column: class")

    if df.empty:
        raise ValueError(f"{name}: dataset is empty.")

    if df[TARGET].isna().any():
        raise ValueError(f"{name}: missing target labels.")

    labels = set(df[TARGET].unique())
    if not labels.issubset({NEGATIVE_LABEL, POSITIVE_LABEL}):
        raise ValueError(f"{name}: unexpected target labels: {labels}")

    return True


def validate_scania_data(train, test):
    """Check dataset integrity and schema consistency."""

    if TARGET not in train.columns or TARGET not in test.columns:
        raise ValueError("Missing target column: class")

    if list(train.columns) != list(test.columns):
        raise ValueError("Training and test schemas do not match.")

    if train.empty or test.empty:
        raise ValueError("One or both datasets are empty.")

    expected_labels = {NEGATIVE_LABEL, POSITIVE_LABEL}

    for name, df in [("Training", train), ("Test", test)]:
        labels = set(df[TARGET].dropna().unique())

        if not labels.issubset(expected_labels):
            raise ValueError(f"{name}: unexpected target labels: {labels}")

        if df[TARGET].isna().any():
            raise ValueError(f"{name}: missing target labels.")

    return True


def split_features_target(df):
    """Split a dataframe into (X, y) with y encoded as pos=1 / neg=0."""

    if TARGET not in df.columns:
        raise ValueError("Missing target column: class")

    y = (df[TARGET] == POSITIVE_LABEL).astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def dataset_summary(train, test):
    """Generate a basic data quality summary."""

    results = {}

    for name, df in [("Training", train), ("Test", test)]:
        results[name] = {
            "rows": len(df),
            "features": len(df.columns) - 1,
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "class_distribution": df[TARGET].value_counts().to_dict(),
        }

    return results
