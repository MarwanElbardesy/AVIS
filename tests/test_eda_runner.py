import json
import re
from pathlib import Path

import pytest

from src.data import scania
from src.eda import run_eda as runner
from src.eda.utils import dumps_deterministic, sha256_file, to_jsonable
from tests.conftest import write_aps_csv
from tests.eda_helpers import make_eda_frame

import numpy as np
import pandas as pd


@pytest.fixture
def train_csv(tmp_path):
    X, y = make_eda_frame(n=800, seed=5)
    df = X.copy()
    df.insert(0, "class", np.where(y == 1, "pos", "neg"))
    return write_aps_csv(df, tmp_path / "train.csv")


def test_runner_writes_results_and_metadata(train_csv, tmp_path):
    out = tmp_path / "out"
    results = runner.run_eda(train_csv, out, seed=1)
    saved = json.loads((out / "eda_results.json").read_text())

    assert saved["metadata"]["train_file_sha256"] == sha256_file(train_csv)
    assert saved["metadata"]["rows"] == 800
    assert saved["metadata"]["seed"] == 1
    assert set(saved["sections"]) == set(runner.SECTIONS)
    assert {"class_balance", "schema", "quality", "missingness"} <= set(saved["sections"])
    assert results["metadata"]["sections"] == list(runner.SECTIONS)
    assert "pandas" in saved["metadata"]["libraries"]


def test_results_are_deterministic_byte_for_byte(train_csv, tmp_path):
    runner.run_eda(train_csv, tmp_path / "a", seed=3)
    runner.run_eda(train_csv, tmp_path / "b", seed=3)
    assert (tmp_path / "a" / "eda_results.json").read_bytes() == (tmp_path / "b" / "eda_results.json").read_bytes()


def test_json_is_strict_no_nan_tokens(train_csv, tmp_path):
    runner.run_eda(train_csv, tmp_path / "o")
    text = (tmp_path / "o" / "eda_results.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)  # strict parse succeeds


def test_runner_never_needs_test_file(train_csv, tmp_path, monkeypatch):
    monkeypatch.setattr(scania, "TEST_FILE", tmp_path / "must_not_be_read.csv")
    runner.run_eda(train_csv, tmp_path / "o")  # would raise if it tried to load it


def test_eda_source_never_references_the_test_set():
    """Static guard: no EDA module may import or mention the test loader/path."""
    forbidden = re.compile(r"load_scania_test|load_scania_data|TEST_FILE|aps_failure_test")
    for path in Path(runner.__file__).parent.glob("*.py"):
        assert not forbidden.search(path.read_text()), f"{path.name} references the test set"


def test_section_subset_and_unknown_section(train_csv, tmp_path):
    results = runner.run_eda(train_csv, tmp_path / "o", sections=["schema"])
    assert list(results["sections"]) == ["schema"]
    with pytest.raises(ValueError):
        runner.run_eda(train_csv, tmp_path / "o2", sections=["nope"])


def test_planted_findings_survive_the_full_pipeline(train_csv, tmp_path):
    q = runner.run_eda(train_csv, tmp_path / "o")["sections"]["quality"]
    assert q["duplicate_column_groups"] == [["aa_000", "ad_000"]]
    assert "cd_000" in q["constant_columns"]
    assert q["duplicate_rows"] == 0


def test_summary_text_mentions_key_facts(train_csv, tmp_path):
    text = runner.format_summary(runner.run_eda(train_csv, tmp_path / "o"))
    assert "800 rows" in text and "histogram candidates: ag, ay" in text and "sha256" in text


def test_to_jsonable_handles_numpy_and_nan():
    obj = {"a": np.int64(3), "b": np.float64("nan"), "c": np.array([1.0, np.inf]), "d": {np.str_("x"): np.bool_(True)}}
    assert to_jsonable(obj) == {"a": 3, "b": None, "c": [1.0, None], "d": {"x": True}}
    assert dumps_deterministic({"b": 1, "a": 2}).index('"a"') < dumps_deterministic({"b": 1, "a": 2}).index('"b"')


def test_sha256_changes_when_data_changes(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a")
    first = sha256_file(f)
    f.write_text("b")
    assert sha256_file(f) != first


def test_all_eight_sections_run_and_are_json_clean(train_csv, tmp_path):
    results = runner.run_eda(train_csv, tmp_path / "o", seed=2)
    expected = {"class_balance", "schema", "quality", "missingness",
                "distributions", "relationships", "signal", "leakage"}
    assert set(results["sections"]) == expected == set(runner.SECTIONS)
    saved = json.loads((tmp_path / "o" / "eda_results.json").read_text())
    assert saved["sections"]["relationships"]["histogram_groups"].keys() == {"ag", "ay"}
    assert "flags" in saved["sections"]["leakage"]


def test_leakage_alone_computes_its_own_signal_table(train_csv, tmp_path):
    results = runner.run_eda(train_csv, tmp_path / "o", sections=["leakage"])
    assert list(results["sections"]) == ["leakage"]
    assert set(results["sections"]["leakage"]["flags"])


def test_signal_is_computed_once_and_shared(train_csv, tmp_path, monkeypatch):
    calls = []
    real = runner.signal.signal_report
    monkeypatch.setattr(runner.signal, "signal_report", lambda *a, **k: calls.append(1) or real(*a, **k))
    runner.run_eda(train_csv, tmp_path / "o", sections=["signal", "leakage"])
    assert len(calls) == 1


def test_summary_covers_new_sections(train_csv, tmp_path):
    text = runner.format_summary(runner.run_eda(train_csv, tmp_path / "o"))
    for needle in ("distributions:", "relationships:", "histogram ag:", "histogram ay:",
                   "signal (descriptive only", "leakage/ordering:"):
        assert needle in text, needle


def test_summary_shows_direction_split_and_tolerates_older_results(train_csv, tmp_path):
    results = runner.run_eda(train_csv, tmp_path / "o", sections=["class_balance", "missingness"])
    text = runner.format_summary(results)
    assert "significant AND notable" in text and "higher /" in text

    ta = results["sections"]["missingness"]["target_association"]
    for key in [k for k in ta if k.startswith(("n_significant_h", "n_significant_l", "n_notable_h", "n_notable_l", "n_significant_and"))]:
        del ta[key]                                       # results written before 2d-1
    assert "significant AND notable" not in runner.format_summary(results)
