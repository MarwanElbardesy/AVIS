import pandas as pd
import pytest

from src.data import scania
from tests.conftest import make_aps_like, write_aps_csv


def test_header_detected_with_license_block(tmp_path):
    df = make_aps_like(n=50)
    path = write_aps_csv(df, tmp_path / "train.csv")
    assert scania.find_header_row(path) == 20


def test_header_detected_with_different_header_length(tmp_path):
    df = make_aps_like(n=50)
    path = write_aps_csv(df, tmp_path / "train.csv", header="a\nb\nc\n")
    loaded = scania.read_aps_csv(path)
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == 50


def test_na_parsed_as_nan(tmp_path):
    df = make_aps_like(n=200)
    path = write_aps_csv(df, tmp_path / "train.csv")
    loaded = scania.read_aps_csv(path)
    assert loaded.isna().sum().sum() == df.isna().sum().sum() > 0


def test_load_train_never_needs_test_file(tmp_path, monkeypatch):
    path = write_aps_csv(make_aps_like(n=60), tmp_path / "train.csv")
    monkeypatch.setattr(scania, "TEST_FILE", tmp_path / "does_not_exist.csv")
    train = scania.load_scania_train(path)
    assert len(train) == 60


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scania.load_scania_train(tmp_path / "nope.csv")


def test_validate_frame_rejects_bad_labels():
    df = pd.DataFrame({"class": ["neg", "bad"], "a": [1, 2]})
    with pytest.raises(ValueError):
        scania.validate_scania_frame(df)


def test_validate_frame_rejects_missing_labels():
    df = pd.DataFrame({"class": ["neg", None], "a": [1, 2]})
    with pytest.raises(ValueError):
        scania.validate_scania_frame(df)


def test_split_features_target_encoding(aps_df):
    X, y = scania.split_features_target(aps_df)
    assert "class" not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert int(y.sum()) == int((aps_df["class"] == "pos").sum())
