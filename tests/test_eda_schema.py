import pytest

from src.eda.schema import describe_groups, group_features, parse_feature_name
from tests.eda_helpers import make_eda_frame


@pytest.mark.parametrize(
    "name, expected",
    [("aa_000", ("aa", 0)), ("ag_009", ("ag", 9)), ("cd_000", ("cd", 0)), ("ee_123", ("ee", 123))],
)
def test_parse_valid_names(name, expected):
    assert parse_feature_name(name) == expected


@pytest.mark.parametrize("name", ["class", "aa", "aa_", "_000", "aa_00x", "a1_000", "aa_000_1"])
def test_parse_invalid_names(name):
    assert parse_feature_name(name) is None


def test_group_features_orders_by_numeric_suffix():
    groups, unparsed = group_features(["ag_010", "ag_002", "ag_001", "zz_000", "class"])
    assert groups["ag"] == ["ag_001", "ag_002", "ag_010"]  # numeric, not lexicographic
    assert unparsed == ["class"]


def test_histogram_candidates_detected_structurally():
    X, _ = make_eda_frame(n=50)
    info = describe_groups(list(X.columns))
    assert info["histogram_candidate_prefixes"] == ["ag", "ay"]
    assert info["groups"]["bb"]["histogram_candidate"] is False        # only 3 bins
    assert info["groups"]["bc"]["n_features"] == 10
    assert info["groups"]["bc"]["histogram_candidate"] is False        # 10 columns but not 0..9
    assert info["unparsed_columns"] == []


def test_custom_bin_count():
    info = describe_groups([f"xx_{i:03d}" for i in range(4)], bins=4)
    assert info["histogram_candidate_prefixes"] == ["xx"]


def test_schema_reads_names_only():
    # Passing bare names (no data at all) must be enough: schema cannot touch values.
    info = describe_groups(["aa_000", "aa_001"])
    assert info["n_groups"] == 1
