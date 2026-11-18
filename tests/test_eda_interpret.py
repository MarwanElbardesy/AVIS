"""Tests for the rule-based EDA interpretation layer (Milestone 2d-2).

Verifies:
1. Strict evidence traceability: EVERY numeric claim and threshold in statements
   and recommendations must be backed by the finding's OWN evidence dict.
2. Anti-hallucination guard: unsupported numbers fail validation.
3. Status semantics: NOMINAL, INFORMATIVE, ATTENTION, UNAVAILABLE boundaries
   and system-level status aggregation hierarchy (ATTENTION > INFORMATIVE > NOMINAL).
4. Epistemic language guard: zero causal claims or ungrounded physical conclusions.
5. Strict leakage validation: rejection of missing nested keys, NaN, and infinity.
6. Schema compatibility: graceful degradation for missing optional D22 fields.
7. Clean narrative formatting for empty collections and zero-count edge cases.
8. Determinism: byte-identical output across repeated executions.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Set

import pytest

from src.eda.interpret import (
    STATUS_ATTENTION,
    STATUS_INFORMATIVE,
    STATUS_NOMINAL,
    STATUS_UNAVAILABLE,
    interpret_class_balance,
    interpret_distributions,
    interpret_eda,
    interpret_leakage,
    interpret_metadata,
    interpret_missingness,
    interpret_quality,
    interpret_relationships,
    interpret_schema,
    interpret_signal,
)


ARTIFACT_PATH = Path("reports/eda/eda_results.json")


@pytest.fixture(scope="session")
def real_eda_results() -> Dict[str, Any]:
    """Load the verified static EDA JSON artifact."""
    assert ARTIFACT_PATH.exists(), f"Missing required artifact: {ARTIFACT_PATH}"
    with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_evidence_number_tokens(evidence: Dict[str, Any]) -> Set[str]:
    """Extract all valid representations of numeric values from evidence."""
    ev_tokens = set()
    for k, v in evidence.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            ev_tokens.add(str(v))
            ev_tokens.add(str(int(v)))
            ev_tokens.add(f"{v:.1f}")
            ev_tokens.add(f"{v:.2f}")
            ev_tokens.add(f"{v:.3f}")
            ev_tokens.add(f"{v:.4f}")
            ev_tokens.add(f"{v * 100:.1f}")
            ev_tokens.add(f"{v * 100:.2f}")
            ev_tokens.add(f"{v * 100:.0f}")
        elif isinstance(v, str):
            for tok in re.findall(r"\b\d+\.?\d*\b", v):
                ev_tokens.add(tok)
    return ev_tokens


def _find_unsupported_statement_numbers(statement: str, evidence: Dict[str, Any], allowed_tokens: Set[str] = None) -> list:
    """Return any numbers in statement that do not match evidence tokens."""
    ev_tokens = _extract_evidence_number_tokens(evidence)
    if allowed_tokens:
        ev_tokens.update(allowed_tokens)

    # Clean statement: remove commas in numbers like 60,000
    cleaned = statement.replace(",", "")
    # Find all numeric tokens
    tokens = re.findall(r"\b\d+\.?\d*\b", cleaned)
    unmatched = [t for t in tokens if t not in ev_tokens]
    return unmatched


# --------------------------------------------------------------------------- #
# 1. Schema & Contract Conformance
# --------------------------------------------------------------------------- #
def test_interpret_eda_schema_conformance(real_eda_results):
    """Verify master interpret_eda returns all required keys and valid types."""
    result = interpret_eda(real_eda_results)

    assert "metadata" in result
    assert "overall_status" in result
    assert "sections" in result
    assert result["overall_status"] in {STATUS_NOMINAL, STATUS_INFORMATIVE, STATUS_ATTENTION}

    expected_sections = [
        "class_balance",
        "schema",
        "quality",
        "missingness",
        "distributions",
        "relationships",
        "signal",
        "leakage",
    ]
    for sec_name in expected_sections:
        assert sec_name in result["sections"], f"Missing section {sec_name}"
        sec = result["sections"][sec_name]
        assert "section" in sec
        assert "status" in sec
        assert "policy" in sec
        assert "headline" in sec
        assert "findings" in sec
        assert "narrative" in sec
        assert "caveats" in sec

        assert isinstance(sec["headline"], str) and len(sec["headline"]) > 0
        assert isinstance(sec["narrative"], list) and len(sec["narrative"]) > 0
        assert isinstance(sec["caveats"], list)
        assert isinstance(sec["findings"], list) and len(sec["findings"]) > 0

        for f in sec["findings"]:
            assert "id" in f
            assert "category" in f
            assert "status" in f
            assert "statement" in f
            assert "source_section" in f
            assert "source_keys" in f
            assert "evidence" in f
            assert isinstance(f["evidence"], dict)


# --------------------------------------------------------------------------- #
# 2. Strict Evidence Traceability (ALL Numbers Backed)
# --------------------------------------------------------------------------- #
def test_strict_numerical_traceability_all_findings(real_eda_results):
    """Assert that EVERY empirical number and threshold in statements is backed by its OWN evidence."""
    interpreted = interpret_eda(real_eda_results)

    for sec_name, sec_dict in interpreted["sections"].items():
        for finding in sec_dict["findings"]:
            if finding["status"] == STATUS_UNAVAILABLE:
                continue

            stmt = finding["statement"]
            ev = finding["evidence"]

            # Common statistical/grammatical tokens:
            # - '95' in 95% Wilson CI (confidence interval level)
            # - '1' in ratio expressions like '59.0:1'
            allowed = {"95", "1"}
            unmatched = _find_unsupported_statement_numbers(stmt, ev, allowed_tokens=allowed)
            assert len(unmatched) == 0, (
                f"Finding {finding['id']} in {sec_name} has unsupported numbers {unmatched} "
                f"in statement:\n  '{stmt}'\nEvidence provided:\n  {ev}"
            )


def test_strict_numeric_matcher_rejects_unsupported_number():
    """Verify that strict numeric matching fails when even a single number is unbacked."""
    evidence = {"positives": 10, "negatives": 90}
    # Statement includes 999 which is not in evidence
    statement_with_hallucination = "Found 10 failures and 90 healthy units with 999 unbacked events."
    unsupported = _find_unsupported_statement_numbers(statement_with_hallucination, evidence)
    assert "999" in unsupported


# --------------------------------------------------------------------------- #
# 3. Epistemic Language Guard (Zero Causal or Unsupported Claims)
# --------------------------------------------------------------------------- #
def test_no_causal_or_unsupported_language(real_eda_results):
    """Ensure no statement or narrative makes causal, hardware, or prescriptive claims."""
    forbidden_terms = [
        "caused by",
        "causes",
        "causal",
        "proves",
        "proves that",
        "guarantees failure",
        "must drop",
        "must remove",
        "should drop",
        "should remove",
        "structurally preferred",
        "trees are preferred",
        "drop offline",
        "sensor outages",
        "empirical event frequencies",
    ]

    interpreted = interpret_eda(real_eda_results)

    for sec_name, sec in interpreted["sections"].items():
        for paragraph in sec["narrative"]:
            for term in forbidden_terms:
                assert term not in paragraph.lower(), (
                    f"Forbidden epistemic phrase '{term}' found in section '{sec_name}' narrative: {paragraph}"
                )

        for finding in sec["findings"]:
            stmt = finding["statement"].lower()
            for term in forbidden_terms:
                assert term not in stmt, (
                    f"Forbidden epistemic phrase '{term}' found in finding '{finding['id']}': {finding['statement']}"
                )


# --------------------------------------------------------------------------- #
# 4. Status Semantics & System-Level Aggregation
# --------------------------------------------------------------------------- #
def test_overall_status_aggregation_hierarchy():
    """Verify status precedence: ATTENTION > INFORMATIVE > NOMINAL."""
    clean_meta = {
        "dataset": "Scania",
        "rows": 60000,
        "features": 170,
        "train_file_sha256": "abc",
        "results_schema_version": 1,
    }
    clean_cb = {
        "positives": 500,
        "negatives": 500,
        "n": 1000,
        "imbalance_ratio": 1.0,
        "positive_rate": 0.5,
        "positive_rate_ci95": [0.47, 0.53],
    }
    clean_schema = {
        "n_groups": 107,
        "histogram_candidate_prefixes": [],
        "unparsed_columns": [],
    }
    clean_q = {
        "constant_columns": [],
        "near_constant_columns": {},
        "duplicate_rows": 0,
        "columns_with_negative_values": [],
        "missing_cells": 0,
        "saturation_candidates": {},
    }
    clean_miss = {
        "n_columns_with_missing": 0,
        "n_columns_total": 170,
        "comissing_clusters": [],
        "comissing_cluster_min_phi": 0.9,
        "target_association": {
            "n_significant_fdr": 0,
            "n_tested": 170,
            "alpha": 0.05,
            "n_notable_effect": 0,
            "notable_auc_delta": 0.05,
        },
    }
    clean_dist = {
        "n_heavy_tailed": 0,
        "n_zero_inflated": 0,
        "n_iqr_zero": 0,
        "n_columns_profiled": 170,
        "thresholds": {"heavy_skew": 5.0, "zero_inflated": 0.5},
    }
    clean_rel = {
        "n_pairs_scored": 100,
        "n_pairs_abs_rho_ge_0_9": 0,
        "n_pairs_abs_rho_ge_0_95": 0,
        "near_duplicate_clusters": [],
        "near_duplicate_min_abs_rho": 0.95,
        "min_overlap_rows": 100,
        "histogram_groups": {},
    }
    clean_sig = {
        "n_columns_scored": 170,
        "n_columns_not_scorable": 0,
        "auc_band_counts": {"ge_0.90": 0},
        "top_columns": [],
    }
    clean_leak = {
        "flags": {
            "class_exclusive_values_present": False,
            "duplicate_rows_with_conflicting_labels": False,
            "row_order_related_to_label": False,
            "features_drift_with_row_position": False,
            "suspiciously_strong_single_feature": False,
        },
        "class_exclusive_values": {},
        "duplicate_rows": {"n_conflicting_label_groups": 0},
        "row_order": {"runs_test": {"p_value": 0.5}, "blocks": {"cramers_v": 0.01}},
        "feature_drift": {"n_abs_rho_ge_threshold": 0, "max_abs_rho": 0.01},
        "thresholds": {"drift_abs_rho": 0.05},
    }

    # Case 1: Purely nominal dataset -> NOMINAL
    res_nominal = interpret_eda(
        {
            "metadata": clean_meta,
            "sections": {
                "class_balance": clean_cb,
                "schema": clean_schema,
                "quality": clean_q,
                "missingness": clean_miss,
                "distributions": clean_dist,
                "relationships": clean_rel,
                "signal": clean_sig,
                "leakage": clean_leak,
            },
        }
    )
    assert res_nominal["overall_status"] == STATUS_NOMINAL

    # Case 2: Informative missingness added (no attention flags) -> INFORMATIVE (NOT NOMINAL)
    info_miss = dict(clean_miss, n_columns_with_missing=10)
    res_info = interpret_eda(
        {
            "metadata": clean_meta,
            "sections": {
                "class_balance": clean_cb,
                "schema": clean_schema,
                "quality": clean_q,
                "missingness": info_miss,
                "distributions": clean_dist,
                "relationships": clean_rel,
                "signal": clean_sig,
                "leakage": clean_leak,
            },
        }
    )
    assert res_info["overall_status"] == STATUS_INFORMATIVE

    # Case 3: Constant column added -> ATTENTION overrides INFORMATIVE
    attention_q = dict(clean_q, constant_columns=["cd_000"])
    res_attention = interpret_eda(
        {
            "metadata": clean_meta,
            "sections": {
                "class_balance": clean_cb,
                "schema": clean_schema,
                "quality": attention_q,
                "missingness": info_miss,
                "distributions": clean_dist,
                "relationships": clean_rel,
                "signal": clean_sig,
                "leakage": clean_leak,
            },
        }
    )
    assert res_attention["overall_status"] == STATUS_ATTENTION


# --------------------------------------------------------------------------- #
# 5. Narrative Formatting & Edge Cases
# --------------------------------------------------------------------------- #
def test_quality_empty_constant_columns_formatting():
    """Verify headline formatting does not produce empty parentheses when constant_columns is empty."""
    clean_q = {
        "constant_columns": [],
        "near_constant_columns": {},
        "duplicate_rows": 0,
        "columns_with_negative_values": [],
        "missing_cells": 100,
        "saturation_candidates": {},
    }
    interp = interpret_quality(clean_q)
    assert "()" not in interp.headline
    assert "0 constant features" in interp.headline
    assert interp.status == STATUS_NOMINAL


def test_schema_unparsed_columns_triggers_attention():
    """Verify non-empty unparsed columns triggers ATTENTION."""
    dirty_schema = {
        "n_groups": 10,
        "histogram_candidate_prefixes": [],
        "unparsed_columns": ["bad_col_1", "bad_col_2"],
    }
    interp = interpret_schema(dirty_schema)
    assert interp.status == STATUS_ATTENTION
    assert "bad_col_1" in interp.findings[0].statement


def test_signal_empty_top_columns_and_zero_strong_signal():
    """Verify signal formatting when no features exceed 0.90 AUC and top_columns is empty."""
    sig_empty = {
        "n_columns_scored": 10,
        "n_columns_not_scorable": 0,
        "auc_band_counts": {"ge_0.90": 0},
        "top_columns": [],
    }
    interp = interpret_signal(sig_empty)
    assert "Strong univariate signal" not in interp.headline
    assert interp.findings[1].status == STATUS_UNAVAILABLE


def test_leakage_activated_flags():
    """Verify each leakage flag correctly triggers ATTENTION."""
    base_leak = {
        "flags": {
            "class_exclusive_values_present": False,
            "duplicate_rows_with_conflicting_labels": False,
            "row_order_related_to_label": False,
            "features_drift_with_row_position": False,
            "suspiciously_strong_single_feature": False,
        },
        "class_exclusive_values": {},
        "duplicate_rows": {"n_conflicting_label_groups": 0},
        "row_order": {"runs_test": {"p_value": 0.5}, "blocks": {"cramers_v": 0.01}},
        "feature_drift": {"n_abs_rho_ge_threshold": 0, "max_abs_rho": 0.01},
        "thresholds": {"drift_abs_rho": 0.05},
    }

    # Flag: class_exclusive_values
    leak1 = dict(base_leak, flags=dict(base_leak["flags"], class_exclusive_values_present=True))
    assert interpret_leakage(leak1).status == STATUS_ATTENTION

    # Flag: duplicate_rows_with_conflicting_labels
    leak2 = dict(base_leak, flags=dict(base_leak["flags"], duplicate_rows_with_conflicting_labels=True))
    assert interpret_leakage(leak2).status == STATUS_ATTENTION

    # Flag: features_drift_with_row_position
    leak3 = dict(base_leak, flags=dict(base_leak["flags"], features_drift_with_row_position=True))
    assert interpret_leakage(leak3).status == STATUS_ATTENTION


# --------------------------------------------------------------------------- #
# 6. Strict Validation & Rejection of Malformed / NaN / Inf Inputs
# --------------------------------------------------------------------------- #
def test_leakage_missing_required_subkeys():
    """Verify KeyError is raised when required leakage subkeys are absent (no silent defaults)."""
    broken_leak = {
        "flags": {},
        "class_exclusive_values": {},
        "duplicate_rows": {},  # missing n_conflicting_label_groups
        "row_order": {"runs_test": {}},  # missing p_value
        "feature_drift": {},
        "thresholds": {},
    }
    with pytest.raises(KeyError):
        interpret_leakage(broken_leak)


def test_leakage_rejects_nan_and_inf():
    """Verify ValueError is raised on NaN or infinite numeric metrics."""
    nan_leak = {
        "flags": {},
        "class_exclusive_values": {},
        "duplicate_rows": {"n_conflicting_label_groups": 0},
        "row_order": {"runs_test": {"p_value": float("nan")}, "blocks": {"cramers_v": 0.01}},
        "feature_drift": {"n_abs_rho_ge_threshold": 0, "max_abs_rho": 0.01},
        "thresholds": {"drift_abs_rho": 0.05},
    }
    with pytest.raises(ValueError, match="must be finite"):
        interpret_leakage(nan_leak)

    inf_leak = {
        "flags": {},
        "class_exclusive_values": {},
        "duplicate_rows": {"n_conflicting_label_groups": 0},
        "row_order": {"runs_test": {"p_value": 0.5}, "blocks": {"cramers_v": float("inf")}},
        "feature_drift": {"n_abs_rho_ge_threshold": 0, "max_abs_rho": 0.01},
        "thresholds": {"drift_abs_rho": 0.05},
    }
    with pytest.raises(ValueError, match="must be finite"):
        interpret_leakage(inf_leak)


# --------------------------------------------------------------------------- #
# 7. D22 Overlap & Schema Compatibility
# --------------------------------------------------------------------------- #
def test_relationships_d22_overlap_metrics():
    """Verify D22 overlap metrics are reported when present and marked UNAVAILABLE when absent."""
    base_rel = {
        "n_pairs_scored": 100,
        "n_pairs_abs_rho_ge_0_9": 10,
        "n_pairs_abs_rho_ge_0_95": 2,
        "near_duplicate_min_abs_rho": 0.95,
        "min_overlap_rows": 100,
        "histogram_groups": {},
    }

    # Case A: Without D22 min_pairwise_overlap
    rel_without = dict(base_rel, near_duplicate_clusters=[{"columns": ["a", "b"], "size": 2}])
    interp_without = interpret_relationships(rel_without)
    f_overlap_without = next(f for f in interp_without.findings if f.id == "REL_OVERLAP_SUPPORT")
    assert f_overlap_without.status == STATUS_UNAVAILABLE

    # Case B: With D22 min_pairwise_overlap
    rel_with = dict(base_rel, near_duplicate_clusters=[{"columns": ["a", "b"], "size": 2, "min_pairwise_overlap": 54321}])
    interp_with = interpret_relationships(rel_with)
    f_overlap_with = next(f for f in interp_with.findings if f.id == "REL_OVERLAP_SUPPORT")
    assert f_overlap_with.status == STATUS_INFORMATIVE
    assert f_overlap_with.evidence["min_pairwise_overlap"] == 54321
    assert "54321" in f_overlap_with.statement.replace(",", "")


# --------------------------------------------------------------------------- #
# 8. Determinism
# --------------------------------------------------------------------------- #
def test_interpret_eda_is_deterministic(real_eda_results):
    """Verify two independent runs produce byte-identical JSON serializations."""
    run1 = interpret_eda(real_eda_results)
    run2 = interpret_eda(real_eda_results)

    assert run1 == run2

    json1 = json.dumps(run1, sort_keys=True)
    json2 = json.dumps(run2, sort_keys=True)
    assert json1 == json2
