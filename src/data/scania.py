
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "scania"

TRAIN_FILE = DATA_DIR / "aps_failure_training_set.csv"
TEST_FILE = DATA_DIR / "aps_failure_test_set.csv"

TARGET = "class"


def load_scania_data():
    """Load and validate the Scania APS training and test datasets."""

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_FILE}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_FILE}")

    train = pd.read_csv(
        TRAIN_FILE,
        skiprows=20,
        na_values="na"
    )

    test = pd.read_csv(
        TEST_FILE,
        skiprows=20,
        na_values="na"
    )

    validate_scania_data(train, test)

    return train, test


def validate_scania_data(train, test):
    """Check dataset integrity and schema consistency."""

    if TARGET not in train.columns or TARGET not in test.columns:
        raise ValueError("Missing target column: class")

    if list(train.columns) != list(test.columns):
        raise ValueError("Training and test schemas do not match.")

    if train.empty or test.empty:
        raise ValueError("One or both datasets are empty.")

    expected_labels = {"neg", "pos"}

    for name, df in [("Training", train), ("Test", test)]:
        labels = set(df[TARGET].dropna().unique())

        if not labels.issubset(expected_labels):
            raise ValueError(f"{name}: unexpected target labels: {labels}")

        if df[TARGET].isna().any():
            raise ValueError(f"{name}: missing target labels.")

    return True


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
