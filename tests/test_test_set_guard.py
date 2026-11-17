"""Repo-wide guard for the README's claim that development code never touches the test set.

Only src/data/scania.py may define the test loader / path. Everything else under src/ must not
reference it. (The final evaluation, when written, will be a deliberate, reviewed exception.)
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ALLOWED = {SRC / "data" / "scania.py"}
FORBIDDEN = re.compile(r"load_scania_test|load_scania_data|TEST_FILE|aps_failure_test")


def test_no_source_module_references_the_test_set():
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if path not in ALLOWED and FORBIDDEN.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"modules referencing the official test set: {offenders}"


def test_guard_actually_detects_a_reference(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("from src.data.scania import load_scania_test\n")
    assert FORBIDDEN.search(bad.read_text())
    assert SRC.exists() and any(SRC.rglob("*.py"))
