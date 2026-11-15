import pandas as pd
import pytest

from src.data.scania import validate_scania_data, dataset_summary


@pytest.fixture
def sample_data():
    train = pd.DataFrame({
        "class": ["neg", "neg", "pos", "neg"],
        "feature_1": [10, 20, 30, None],
        "feature_2": [1, 2, 3, 4],
    })

    test = pd.DataFrame({
        "class": ["neg", "pos"],
        "feature_1": [15, 25],
        "feature_2": [2, 3],
    })

    return train, test


def test_valid_data(sample_data):
    train, test = sample_data
    assert validate_scania_data(train, test) is True


def test_inconsistent_schema(sample_data):
    train, test = sample_data
    test = test.rename(columns={"feature_2": "feature_3"})

    with pytest.raises(ValueError):
        validate_scania_data(train, test)


def test_missing_target(sample_data):
    train, test = sample_data
    train = train.drop(columns=["class"])

    with pytest.raises(ValueError):
        validate_scania_data(train, test)


def test_invalid_labels(sample_data):
    train, test = sample_data
    train.loc[0, "class"] = "unknown"

    with pytest.raises(ValueError):
        validate_scania_data(train, test)


def test_dataset_summary(sample_data):
    train, test = sample_data
    summary = dataset_summary(train, test)

    assert summary["Training"]["rows"] == 4
    assert summary["Training"]["features"] == 2
    assert summary["Training"]["missing_values"] == 1
    assert summary["Training"]["class_distribution"]["neg"] == 3
