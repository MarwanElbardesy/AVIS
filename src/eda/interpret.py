"""Deterministic, rule-based interpretation layer for Scania APS EDA results.

Converts verified EDA JSON output into evidence-backed, traceable report findings
without statistical recomputation, invented figures, or causal assertions.

Architectural Decisions:
- D12: Reproducible, deterministic results artifact.
- D14: Missingness is descriptive, never prescriptive.
- D15: Effect sizes next to p-values.
- D16: Co-missingness clustering at phi >= 0.9.
- D17: Rank correlation without blind dropping.
- D18: Univariate signal is descriptive only (leakage guard).
- D19: Leakage flags are review prompts, not verdicts; vehicle ID limitation.
- D21: Distribution profiles never transform features automatically.
- D22: Additive fields tolerance and backward compatibility.
- D23: Pure interpretation layer with explicit evidence traceability,
       deterministic status precedence, and strict epistemic guardrails.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Status Semantics
# --------------------------------------------------------------------------- #
# NOMINAL:     Metric or check is within normal expectations; no flags, alerts,
#              or pipeline interventions required.
# INFORMATIVE: Domain observation, statistical structure, or pattern that
#              characterises the dataset without implying a data-quality defect.
# ATTENTION:   Empirical finding requiring explicit action, safeguard, or
#              precaution in downstream modeling, preprocessing, or evaluation.
# UNAVAILABLE: Evidence key was absent from the artifact (e.g. older schema).
STATUS_NOMINAL = "NOMINAL"
STATUS_INFORMATIVE = "INFORMATIVE"
STATUS_ATTENTION = "ATTENTION"
STATUS_UNAVAILABLE = "UNAVAILABLE"

VALID_STATUSES = {STATUS_NOMINAL, STATUS_INFORMATIVE, STATUS_ATTENTION, STATUS_UNAVAILABLE}

# Documented diagnostic threshold for extreme class imbalance (D23)
# Distinct from D5 asymmetric operational cost ratio (50:1 = $500 FN / $10 FP)
IMBALANCE_ATTENTION_RATIO = 10.0

# Documented structural schema constants (D13)
HISTOGRAM_BINS_COUNT = 10
HISTOGRAM_BIN_RANGE = "000 to 009"

# Documented data quality constant (quality.py)
NEAR_CONSTANT_THRESHOLD = 0.999

# Null hypothesis baseline AUC for symmetric two-class indicator association (D15)
NULL_INDICATOR_AUC = 0.5


# --------------------------------------------------------------------------- #
# Data Classes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Finding:
    """An individual evidence-backed finding with exact provenance."""
    id: str
    category: str
    status: str
    statement: str
    source_section: str
    source_keys: List[str]
    evidence: Dict[str, Any]
    recommendation: Optional[str] = None

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{self.status}'. Must be one of {VALID_STATUSES}")
        if not self.source_section:
            raise ValueError("source_section must be non-empty")
        if not self.source_keys:
            raise ValueError("source_keys must specify at least one key")
        if not isinstance(self.evidence, dict):
            raise TypeError("evidence must be a dictionary")


@dataclass(frozen=True)
class SectionInterpretation:
    """Interpreted narrative and findings for an entire EDA section."""
    section: str
    status: str
    policy: str
    headline: str
    findings: List[Finding]
    narrative: List[str]
    caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section": self.section,
            "status": self.status,
            "policy": self.policy,
            "headline": self.headline,
            "findings": [asdict(f) for f in self.findings],
            "narrative": list(self.narrative),
            "caveats": list(self.caveats),
        }


# --------------------------------------------------------------------------- #
# Validation Helpers
# --------------------------------------------------------------------------- #
def _require_finite_float(val: Any, name: str) -> float:
    """Validate that val can be parsed as a finite float (rejecting NaN and Inf)."""
    try:
        fval = float(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Value for '{name}' must be a valid float, got: {val!r}") from e
    if math.isnan(fval) or math.isinf(fval):
        raise ValueError(f"Value for '{name}' must be finite, got: {fval}")
    return fval


def _require_non_negative_int(val: Any, name: str) -> int:
    """Validate that val can be parsed as a non-negative integer."""
    try:
        ival = int(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Value for '{name}' must be a valid integer, got: {val!r}") from e
    if ival < 0:
        raise ValueError(f"Value for '{name}' must be non-negative, got: {ival}")
    return ival


def _aggregate_status(statuses: List[str]) -> str:
    """Aggregate multiple statuses using strict precedence: ATTENTION > INFORMATIVE > NOMINAL > UNAVAILABLE."""
    status_set = set(statuses)
    if STATUS_ATTENTION in status_set:
        return STATUS_ATTENTION
    if STATUS_INFORMATIVE in status_set:
        return STATUS_INFORMATIVE
    if STATUS_NOMINAL in status_set:
        return STATUS_NOMINAL
    return STATUS_UNAVAILABLE


# --------------------------------------------------------------------------- #
# Section Evaluators
# --------------------------------------------------------------------------- #
def interpret_metadata(meta: Dict[str, Any]) -> SectionInterpretation:
    """Validate dataset provenance and scope from metadata."""
    if not isinstance(meta, dict):
        raise TypeError("metadata must be a dictionary")
    required = ["dataset", "rows", "features", "train_file_sha256", "results_schema_version"]
    for k in required:
        if k not in meta:
            raise KeyError(f"Metadata missing required key: {k}")

    rows = _require_non_negative_int(meta["rows"], "metadata.rows")
    features = _require_non_negative_int(meta["features"], "metadata.features")
    sha = str(meta["train_file_sha256"])
    dataset = str(meta["dataset"])
    schema_ver = int(meta["results_schema_version"])

    evidence = {
        "dataset": dataset,
        "rows": rows,
        "features": features,
        "train_file_sha256": sha,
        "results_schema_version": schema_ver,
    }

    f = Finding(
        id="META_DATASET_SCOPE",
        category="PROVENANCE",
        status=STATUS_NOMINAL,
        statement=(
            f"Evaluation is strictly confined to the training dataset ({rows:,} rows, "
            f"{features} features, SHA-256: {sha}). The official test set was not accessed."
        ),
        source_section="metadata",
        source_keys=required,
        evidence=evidence,
        recommendation=None,
    )

    return SectionInterpretation(
        section="metadata",
        status=STATUS_NOMINAL,
        policy="Training dataset only. Test set isolation strictly enforced (D1, D10, D20).",
        headline=f"Analysis verified on {rows:,} training rows across {features} features.",
        findings=[f],
        narrative=[f.statement],
        caveats=["The official test set remains sealed until final pipeline verification."],
    )


def interpret_class_balance(cb: Dict[str, Any]) -> SectionInterpretation:
    """Interpret class distribution and cost-sensitivity implications."""
    if not isinstance(cb, dict):
        raise TypeError("class_balance must be a dictionary")
    required = ["positives", "negatives", "n", "imbalance_ratio", "positive_rate", "positive_rate_ci95"]
    for k in required:
        if k not in cb:
            raise KeyError(f"class_balance missing required key: {k}")

    pos = _require_non_negative_int(cb["positives"], "class_balance.positives")
    neg = _require_non_negative_int(cb["negatives"], "class_balance.negatives")
    n = _require_non_negative_int(cb["n"], "class_balance.n")
    ratio = _require_finite_float(cb["imbalance_ratio"], "class_balance.imbalance_ratio")
    rate = _require_finite_float(cb["positive_rate"], "class_balance.positive_rate")
    ci = cb["positive_rate_ci95"]
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("positive_rate_ci95 must be a 2-element list")
    ci_low = _require_finite_float(ci[0], "class_balance.positive_rate_ci95[0]")
    ci_high = _require_finite_float(ci[1], "class_balance.positive_rate_ci95[1]")

    evidence = {
        "positives": pos,
        "negatives": neg,
        "n": n,
        "imbalance_ratio": ratio,
        "positive_rate": rate,
        "positive_percentage": round(rate * 100, 2),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "ci95_low_percentage": round(ci_low * 100, 2),
        "ci95_high_percentage": round(ci_high * 100, 2),
        "imbalance_threshold": IMBALANCE_ATTENTION_RATIO,
        "cost_fp": 10,
        "cost_fn": 500,
    }

    # Severe class imbalance requires explicit modeling attention (D23)
    status = STATUS_ATTENTION if ratio >= IMBALANCE_ATTENTION_RATIO else STATUS_NOMINAL

    f_imbalance = Finding(
        id="CB_SEVERE_IMBALANCE",
        category="CLASS_DISTRIBUTION",
        status=status,
        statement=(
            f"Failures constitute {rate * 100:.2f}% of training observations "
            f"({pos:,} of {n:,}; 95% Wilson CI: [{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]), "
            f"establishing an imbalance ratio of {ratio:.1f}:1."
        ),
        source_section="class_balance",
        source_keys=required,
        evidence=evidence,
        recommendation=(
            f"Modeling consideration: Standard accuracy is invalid under {ratio:.1f}:1 class imbalance; "
            f"model evaluation and threshold selection must be driven by asymmetric cost "
            f"(10 FP + 500 FN) and PR-AUC rather than default 0.5 decision boundaries (D5)."
        ),
    )

    headline = f"Severe class imbalance observed ({ratio:.1f}:1 negative-to-positive ratio, {rate * 100:.2f}% positives)."
    narrative = [
        f_imbalance.statement,
        (
            f"The training set contains {neg:,} negative instances and {pos:,} positive instances. "
            f"Because positive instances represent {rate * 100:.2f}% of trucks ({pos:,} of {n:,}), "
            f"conventional unweighted loss functions and raw accuracy metrics will fail to detect critical failures."
        ),
    ]

    return SectionInterpretation(
        section="class_balance",
        status=status,
        policy="Cost-aware evaluation mandatory (D5).",
        headline=headline,
        findings=[f_imbalance],
        narrative=narrative,
        caveats=["Observed positive prevalence differs between training (~1.67%) and test (~2.34%) (OM3)."],
    )


def interpret_schema(schema: Dict[str, Any]) -> SectionInterpretation:
    """Interpret structural naming patterns and feature grouping."""
    if not isinstance(schema, dict):
        raise TypeError("schema must be a dictionary")
    required = ["n_groups", "histogram_candidate_prefixes", "unparsed_columns"]
    for k in required:
        if k not in schema:
            raise KeyError(f"schema missing required key: {k}")

    n_groups = _require_non_negative_int(schema["n_groups"], "schema.n_groups")
    hist_prefixes = list(schema["histogram_candidate_prefixes"])
    unparsed = list(schema["unparsed_columns"])

    evidence_groups = {
        "n_groups": n_groups,
        "unparsed_count": len(unparsed),
        "unparsed_columns": unparsed,
    }

    status_groups = STATUS_NOMINAL if len(unparsed) == 0 else STATUS_ATTENTION
    statement_groups = (
        f"All features adhere to the standardized prefix_NNN naming pattern across "
        f"{n_groups} distinct prefix groups with {len(unparsed)} unparsed columns."
        if len(unparsed) == 0
        else f"Schema screening detected {len(unparsed)} unparsed columns ({', '.join(unparsed)}) across {n_groups} groups."
    )

    f_groups = Finding(
        id="SCHEMA_GROUP_CONFORMANCE",
        category="SCHEMA_STRUCTURE",
        status=status_groups,
        statement=statement_groups,
        source_section="schema",
        source_keys=["n_groups", "unparsed_columns"],
        evidence=evidence_groups,
    )

    if len(hist_prefixes) > 0:
        evidence_hist = {
            "candidate_prefixes": sorted(hist_prefixes),
            "n_candidates": len(hist_prefixes),
            "bins_per_histogram": HISTOGRAM_BINS_COUNT,
            "bin_range": HISTOGRAM_BIN_RANGE,
        }
        f_hist = Finding(
            id="SCHEMA_HISTOGRAM_CANDIDATES",
            category="SCHEMA_STRUCTURE",
            status=STATUS_INFORMATIVE,
            statement=(
                f"Structural schema parsing identified {len(hist_prefixes)} candidate histogram "
                f"families ({', '.join(sorted(hist_prefixes))}), each comprising "
                f"{HISTOGRAM_BINS_COUNT} contiguous bin columns ({HISTOGRAM_BIN_RANGE})."
            ),
            source_section="schema",
            source_keys=["histogram_candidate_prefixes"],
            evidence=evidence_hist,
            recommendation=(
                "Modeling consideration: Histogram candidate bins represent contiguous ranges and "
                "should be inspected for bin-order consistency rather than treated as independent sensors (D13)."
            ),
        )
    else:
        evidence_hist = {
            "candidate_prefixes": [],
            "n_candidates": 0,
        }
        f_hist = Finding(
            id="SCHEMA_HISTOGRAM_CANDIDATES",
            category="SCHEMA_STRUCTURE",
            status=STATUS_NOMINAL,
            statement="Structural schema parsing identified 0 candidate histogram families.",
            source_section="schema",
            source_keys=["histogram_candidate_prefixes"],
            evidence=evidence_hist,
            recommendation=None,
        )

    sec_status = _aggregate_status([f_groups.status, f_hist.status])
    headline = f"Schema conforms to standard structure across {n_groups} feature groups with {len(hist_prefixes)} candidate histogram families."
    narrative = [f_groups.statement, f_hist.statement]

    return SectionInterpretation(
        section="schema",
        status=sec_status,
        policy="Structural-only inference from column names (D13).",
        headline=headline,
        findings=[f_groups, f_hist],
        narrative=narrative,
        caveats=["Feature meanings are masked; groupings reflect lexical sensor prefixes, not verified electrical schematics."],
    )


def interpret_quality(qual: Dict[str, Any]) -> SectionInterpretation:
    """Interpret data integrity, zero-variance columns, and saturation."""
    if not isinstance(qual, dict):
        raise TypeError("quality must be a dictionary")
    required = [
        "constant_columns",
        "near_constant_columns",
        "duplicate_rows",
        "columns_with_negative_values",
        "missing_cells",
        "saturation_candidates",
    ]
    for k in required:
        if k not in qual:
            raise KeyError(f"quality missing required key: {k}")

    constants = list(qual["constant_columns"])
    near_constants = qual["near_constant_columns"]
    dup_rows = _require_non_negative_int(qual["duplicate_rows"], "quality.duplicate_rows")
    neg_cols = list(qual["columns_with_negative_values"])
    missing_cells = _require_non_negative_int(qual["missing_cells"], "quality.missing_cells")
    sat_candidates = qual["saturation_candidates"]

    findings = []

    # Constant columns
    has_constants = len(constants) > 0
    f_const = Finding(
        id="QUAL_CONSTANT_COLUMNS",
        category="DATA_QUALITY",
        status=STATUS_ATTENTION if has_constants else STATUS_NOMINAL,
        statement=(
            f"Exactly {len(constants)} constant feature ({', '.join(constants)}) was identified "
            f"with zero variance across all training instances."
            if has_constants
            else "Zero constant features were identified in the training set."
        ),
        source_section="quality",
        source_keys=["constant_columns"],
        evidence={"constant_columns": constants, "count": len(constants)},
        recommendation=(
            f"Modeling consideration: Constant column {', '.join(constants)} provides zero variance "
            f"and must be dropped inside the cross-validation preprocessing pipeline (D4)."
            if has_constants
            else None
        ),
    )
    findings.append(f_const)

    # Duplicate rows
    has_dups = dup_rows > 0
    f_dups = Finding(
        id="QUAL_DUPLICATE_ROWS",
        category="DATA_QUALITY",
        status=STATUS_ATTENTION if has_dups else STATUS_NOMINAL,
        statement=f"Exact duplicate row check identified {dup_rows} duplicate rows across the training set.",
        source_section="quality",
        source_keys=["duplicate_rows"],
        evidence={"duplicate_rows": dup_rows},
    )
    findings.append(f_dups)

    # Value domain constraints
    has_neg = len(neg_cols) > 0
    f_neg = Finding(
        id="QUAL_DOMAIN_SIGNS",
        category="DATA_QUALITY",
        status=STATUS_ATTENTION if has_neg else STATUS_NOMINAL,
        statement=(
            f"{len(neg_cols)} columns contained negative values."
            if has_neg
            else "All observed sensor readings are non-negative; zero columns contained negative values."
        ),
        source_section="quality",
        source_keys=["columns_with_negative_values"],
        evidence={"negative_columns": neg_cols, "count": len(neg_cols)},
    )
    findings.append(f_neg)

    # Near-constant and saturation
    n_near = len(near_constants) if isinstance(near_constants, dict) else len(near_constants)
    n_sat = len(sat_candidates) if isinstance(sat_candidates, dict) else len(sat_candidates)
    has_near_or_sat = (n_near > 0 or n_sat > 0)
    f_near = Finding(
        id="QUAL_NEAR_CONSTANT_AND_SATURATION",
        category="DATA_QUALITY",
        status=STATUS_INFORMATIVE if has_near_or_sat else STATUS_NOMINAL,
        statement=(
            f"Screening identified {n_near} near-constant features (>={NEAR_CONSTANT_THRESHOLD * 100:.1f}% single-value frequency) "
            f"and {n_sat} features exhibiting upper-bound value saturation."
        ),
        source_section="quality",
        source_keys=["near_constant_columns", "saturation_candidates"],
        evidence={
            "n_near_constant": n_near,
            "near_constant_threshold": NEAR_CONSTANT_THRESHOLD,
            "threshold_percentage": round(NEAR_CONSTANT_THRESHOLD * 100, 1),
            "n_saturation_candidates": n_sat,
        },
    )
    findings.append(f_near)

    sec_status = _aggregate_status([f.status for f in findings])

    # Format constant description cleanly without empty parentheses
    if has_constants:
        const_desc = f"{len(constants)} constant feature ({', '.join(constants)})"
    else:
        const_desc = "0 constant features"

    headline = (
        f"Data quality screening identified {const_desc} and {dup_rows} duplicate rows across {missing_cells:,} missing cells."
    )
    narrative = [f.statement for f in findings]

    return SectionInterpretation(
        section="quality",
        status=sec_status,
        policy="Sparse columns are never removed automatically; quality flags report facts (D11).",
        headline=headline,
        findings=findings,
        narrative=narrative,
        caveats=["Near-constant and saturated features are documented but not dropped automatically."],
    )


def interpret_missingness(miss: Dict[str, Any]) -> SectionInterpretation:
    """Interpret missingness distribution, target association, and co-missing clusters."""
    if not isinstance(miss, dict):
        raise TypeError("missingness must be a dictionary")
    required = [
        "n_columns_with_missing",
        "n_columns_total",
        "comissing_clusters",
        "comissing_cluster_min_phi",
        "target_association",
    ]
    for k in required:
        if k not in miss:
            raise KeyError(f"missingness missing required key: {k}")

    n_with = _require_non_negative_int(miss["n_columns_with_missing"], "missingness.n_columns_with_missing")
    n_total = _require_non_negative_int(miss["n_columns_total"], "missingness.n_columns_total")
    clusters = list(miss["comissing_clusters"])
    min_phi = _require_finite_float(miss["comissing_cluster_min_phi"], "missingness.comissing_cluster_min_phi")
    ta = miss["target_association"]
    if not isinstance(ta, dict):
        raise TypeError("missingness target_association must be a dictionary")

    findings = []

    # Overall missingness scope
    share = n_with / n_total if n_total > 0 else 0.0
    f_scope = Finding(
        id="MISS_PERVASIVENESS",
        category="MISSINGNESS_STRUCTURE",
        status=STATUS_INFORMATIVE if n_with > 0 else STATUS_NOMINAL,
        statement=(
            f"Missing data affects {n_with} of {n_total} features ({share * 100:.1f}%), "
            f"indicating that missingness is widespread across features rather than confined to isolated columns."
        ),
        source_section="missingness",
        source_keys=["n_columns_with_missing", "n_columns_total"],
        evidence={
            "n_columns_with_missing": n_with,
            "n_columns_total": n_total,
            "share": share,
            "percentage": round(share * 100, 1),
        },
    )
    findings.append(f_scope)

    # Target association (informative signal, NOT a data defect per D3, D14)
    ta_keys = ["n_significant_fdr", "n_tested", "alpha", "n_notable_effect", "notable_auc_delta"]
    for k in ta_keys:
        if k not in ta:
            raise KeyError(f"target_association missing required key: {k}")

    n_sig = _require_non_negative_int(ta["n_significant_fdr"], "target_association.n_significant_fdr")
    n_tested = _require_non_negative_int(ta["n_tested"], "target_association.n_tested")
    alpha = _require_finite_float(ta["alpha"], "target_association.alpha")
    n_notable = _require_non_negative_int(ta["n_notable_effect"], "target_association.n_notable_effect")
    auc_delta = _require_finite_float(ta["notable_auc_delta"], "target_association.notable_auc_delta")

    has_assoc = (n_sig > 0 or n_notable > 0)
    f_assoc = Finding(
        id="MISS_TARGET_ASSOCIATION",
        category="STATISTICAL_SIGNAL",
        status=STATUS_INFORMATIVE if has_assoc else STATUS_NOMINAL,
        statement=(
            f"Missingness is strongly associated with the failure target: {n_sig} of {n_tested} tested "
            f"features demonstrate statistically significant association under Benjamini-Hochberg FDR "
            f"(alpha = {alpha}), with {n_notable} displaying notable effect sizes "
            f"(|indicator AUC - {NULL_INDICATOR_AUC:.1f}| >= {auc_delta})."
            if has_assoc
            else f"Target association screening identified {n_sig} of {n_tested} features with significant missingness-target association (FDR alpha = {alpha}) and {n_notable} with notable effect sizes (|indicator AUC - {NULL_INDICATOR_AUC:.1f}| >= {auc_delta})."
        ),
        source_section="missingness",
        source_keys=["target_association.n_significant_fdr", "target_association.n_notable_effect"],
        evidence={
            "n_significant_fdr": n_sig,
            "n_tested": n_tested,
            "alpha": alpha,
            "n_notable_effect": n_notable,
            "notable_auc_delta": auc_delta,
            "null_auc": NULL_INDICATOR_AUC,
        },
        recommendation=(
            "Modeling consideration: Because missingness carries strong discriminative signal, "
            "missingness indicators must be preserved in linear/tree pipelines rather than imputed away (D3, D14)."
            if has_assoc
            else None
        ),
    )
    findings.append(f_assoc)

    # D22 Directional breakdown: handle present vs absent explicitly
    d22_keys = [
        "n_significant_higher_missing_in_positives",
        "n_significant_lower_missing_in_positives",
        "n_notable_higher_missing_in_positives",
        "n_notable_lower_missing_in_positives",
        "n_significant_and_notable",
    ]
    has_d22 = all(k in ta for k in d22_keys)
    if has_d22:
        f_dir = Finding(
            id="MISS_DIRECTIONAL_SPLITS",
            category="STATISTICAL_SIGNAL",
            status=STATUS_INFORMATIVE,
            statement=(
                f"Among significant associations, {ta['n_significant_higher_missing_in_positives']} features "
                f"exhibit higher missingness in failure instances while {ta['n_significant_lower_missing_in_positives']} "
                f"exhibit lower missingness in failures ({ta['n_significant_and_notable']} are both significant and notable)."
            ),
            source_section="missingness",
            source_keys=[f"target_association.{k}" for k in d22_keys],
            evidence={k: int(ta[k]) for k in d22_keys},
        )
    else:
        f_dir = Finding(
            id="MISS_DIRECTIONAL_SPLITS",
            category="STATISTICAL_SIGNAL",
            status=STATUS_UNAVAILABLE,
            statement="Directional splits of significant missingness indicators are unavailable in this artifact schema.",
            source_section="missingness",
            source_keys=["target_association"],
            evidence={"available": False, "missing_keys": d22_keys},
        )
    findings.append(f_dir)

    # Co-missingness clusters
    n_clusters = len(clusters)
    largest_size = clusters[0]["size"] if clusters else 0
    f_clust = Finding(
        id="MISS_COMISSING_CLUSTERS",
        category="MISSINGNESS_STRUCTURE",
        status=STATUS_INFORMATIVE if n_clusters > 0 else STATUS_NOMINAL,
        statement=(
            f"Average-linkage clustering at phi >= {min_phi} identifies {n_clusters} co-missing feature clusters, "
            f"led by a primary cluster of {largest_size} features that share an identical missingness pattern."
            if n_clusters > 0
            else f"Average-linkage clustering at phi >= {min_phi} identifies 0 co-missing feature clusters."
        ),
        source_section="missingness",
        source_keys=["comissing_clusters", "comissing_cluster_min_phi"],
        evidence={"n_clusters": n_clusters, "min_phi": min_phi, "largest_cluster_size": largest_size},
    )
    findings.append(f_clust)

    sec_status = _aggregate_status([f.status for f in findings])
    headline = (
        f"Missingness affects {n_with} features with {n_sig} showing significant target association (FDR alpha = {alpha}); "
        f"{n_clusters} co-missing clusters identify correlated missingness patterns."
    )
    narrative = [f.statement for f in findings if f.status != STATUS_UNAVAILABLE]

    return SectionInterpretation(
        section="missingness",
        status=sec_status,
        policy="Descriptive only; missingness is signal, never a reason to delete features (D3, D14).",
        headline=headline,
        findings=findings,
        narrative=narrative,
        caveats=["Missingness rankings cannot be used for global feature selection outside cross-validation folds."],
    )


def interpret_distributions(dist: Dict[str, Any]) -> SectionInterpretation:
    """Interpret empirical shape: heavy tails, zero-inflation, and IQR collapse."""
    if not isinstance(dist, dict):
        raise TypeError("distributions must be a dictionary")
    required = ["n_heavy_tailed", "n_zero_inflated", "n_iqr_zero", "n_columns_profiled", "thresholds"]
    for k in required:
        if k not in dist:
            raise KeyError(f"distributions missing required key: {k}")

    n_heavy = _require_non_negative_int(dist["n_heavy_tailed"], "distributions.n_heavy_tailed")
    n_zero = _require_non_negative_int(dist["n_zero_inflated"], "distributions.n_zero_inflated")
    n_iqr = _require_non_negative_int(dist["n_iqr_zero"], "distributions.n_iqr_zero")
    n_prof = _require_non_negative_int(dist["n_columns_profiled"], "distributions.n_columns_profiled")
    th = dist["thresholds"]
    if not isinstance(th, dict):
        raise TypeError("distributions thresholds must be a dictionary")
    skew_th = _require_finite_float(th.get("heavy_skew", 5.0), "thresholds.heavy_skew")
    zero_th = _require_finite_float(th.get("zero_inflated", 0.5), "thresholds.zero_inflated")

    findings = []

    # Heavy tails
    share_heavy = n_heavy / n_prof if n_prof > 0 else 0.0
    has_heavy = n_heavy > 0
    f_heavy = Finding(
        id="DIST_HEAVY_TAILS",
        category="DISTRIBUTION_SHAPE",
        status=STATUS_INFORMATIVE if has_heavy else STATUS_NOMINAL,
        statement=(
            f"{n_heavy} of {n_prof} profiled features ({share_heavy * 100:.1f}%) exhibit "
            f"heavy-tailed distributions with extreme sample skewness (|skew| >= {skew_th:.1f})."
        ),
        source_section="distributions",
        source_keys=["n_heavy_tailed", "n_columns_profiled", "thresholds.heavy_skew"],
        evidence={
            "n_heavy_tailed": n_heavy,
            "n_profiled": n_prof,
            "heavy_skew_threshold": skew_th,
            "share": share_heavy,
            "percentage": round(share_heavy * 100, 1),
        },
        recommendation=(
            "Modeling consideration: Given pervasive heavy tails, tree-based models or non-linear "
            "quantile transformations should be evaluated against raw parametric estimators in CV (D21)."
            if has_heavy
            else None
        ),
    )
    findings.append(f_heavy)

    # Zero inflation
    f_zero = Finding(
        id="DIST_ZERO_INFLATION",
        category="DISTRIBUTION_SHAPE",
        status=STATUS_INFORMATIVE if n_zero > 0 else STATUS_NOMINAL,
        statement=(
            f"{n_zero} features are zero-inflated, having zero values in at least {zero_th * 100:.0f}% of observed rows."
        ),
        source_section="distributions",
        source_keys=["n_zero_inflated", "thresholds.zero_inflated"],
        evidence={
            "n_zero_inflated": n_zero,
            "zero_inflated_threshold": zero_th,
            "percentage_threshold": round(zero_th * 100, 1),
        },
    )
    findings.append(f_zero)

    # IQR zero
    f_iqr = Finding(
        id="DIST_IQR_COLLAPSE",
        category="DISTRIBUTION_SHAPE",
        status=STATUS_INFORMATIVE if n_iqr > 0 else STATUS_NOMINAL,
        statement=(
            f"{n_iqr} features exhibit an interquartile range of zero (Q1 = Q3), causing "
            f"standard Tukey boxplot outlier fences to collapse onto the median."
        ),
        source_section="distributions",
        source_keys=["n_iqr_zero"],
        evidence={"n_iqr_zero": n_iqr},
    )
    findings.append(f_iqr)

    sec_status = _aggregate_status([f.status for f in findings])
    if n_heavy > 0 or n_zero > 0:
        headline = f"Pervasive heavy tails ({n_heavy} features with |skew| >= {skew_th}) and {n_zero} zero-inflated features."
    else:
        headline = f"Distribution profiling across {n_prof} features identified 0 heavy-tailed and 0 zero-inflated features."
    narrative = [f.statement for f in findings]

    return SectionInterpretation(
        section="distributions",
        status=sec_status,
        policy="Descriptive only; distribution profiles never transform or remove features automatically (D21).",
        headline=headline,
        findings=findings,
        narrative=narrative,
        caveats=["Skewness and zero-inflation statistics describe raw observed data without implying data invalidity."],
    )


def interpret_relationships(rel: Dict[str, Any]) -> SectionInterpretation:
    """Interpret rank correlation, near-duplicate clusters, and histogram groups."""
    if not isinstance(rel, dict):
        raise TypeError("relationships must be a dictionary")
    required = [
        "n_pairs_scored",
        "n_pairs_abs_rho_ge_0_9",
        "n_pairs_abs_rho_ge_0_95",
        "near_duplicate_clusters",
        "near_duplicate_min_abs_rho",
        "min_overlap_rows",
        "histogram_groups",
    ]
    for k in required:
        if k not in rel:
            raise KeyError(f"relationships missing required key: {k}")

    n_scored = _require_non_negative_int(rel["n_pairs_scored"], "relationships.n_pairs_scored")
    n_90 = _require_non_negative_int(rel["n_pairs_abs_rho_ge_0_9"], "relationships.n_pairs_abs_rho_ge_0_9")
    n_95 = _require_non_negative_int(rel["n_pairs_abs_rho_ge_0_95"], "relationships.n_pairs_abs_rho_ge_0_95")
    clusters = list(rel["near_duplicate_clusters"])
    min_rho = _require_finite_float(rel["near_duplicate_min_abs_rho"], "relationships.near_duplicate_min_abs_rho")
    min_overlap = _require_non_negative_int(rel["min_overlap_rows"], "relationships.min_overlap_rows")
    hist_groups = rel["histogram_groups"]

    findings = []

    # Collinearity
    has_pairs = (n_90 > 0 or n_95 > 0)
    f_pairs = Finding(
        id="REL_MULTICOLLINEARITY",
        category="FEATURE_RELATIONSHIPS",
        status=STATUS_INFORMATIVE if has_pairs else STATUS_NOMINAL,
        statement=(
            f"Pairwise rank correlation across {n_scored:,} valid feature pairs (min overlap {min_overlap} rows) "
            f"identified {n_90} pairs with |rho| >= {0.90:.2f} and {n_95} pairs with |rho| >= {0.95:.2f}."
        ),
        source_section="relationships",
        source_keys=["n_pairs_scored", "n_pairs_abs_rho_ge_0_9", "n_pairs_abs_rho_ge_0_95", "min_overlap_rows"],
        evidence={
            "n_pairs_scored": n_scored,
            "n_pairs_abs_rho_ge_0_9": n_90,
            "n_pairs_abs_rho_ge_0_95": n_95,
            "threshold_high": 0.90,
            "threshold_extreme": 0.95,
            "min_overlap_rows": min_overlap,
        },
    )
    findings.append(f_pairs)

    # Near-duplicate clusters
    n_clusters = len(clusters)
    largest_size = clusters[0]["size"] if clusters else 0
    f_clust = Finding(
        id="REL_NEAR_DUPLICATE_CLUSTERS",
        category="FEATURE_RELATIONSHIPS",
        status=STATUS_INFORMATIVE if n_clusters > 0 else STATUS_NOMINAL,
        statement=(
            f"Hierarchical clustering at |rho| >= {min_rho} isolated {n_clusters} near-duplicate feature clusters, "
            f"led by a {largest_size}-feature cluster."
            if n_clusters > 0
            else f"Hierarchical clustering at |rho| >= {min_rho} isolated 0 near-duplicate feature clusters."
        ),
        source_section="relationships",
        source_keys=["near_duplicate_clusters", "near_duplicate_min_abs_rho"],
        evidence={"n_clusters": n_clusters, "min_abs_rho": min_rho, "largest_cluster_size": largest_size},
        recommendation=(
            "Modeling consideration: Highly correlated clusters represent redundancy, not invalidity. "
            "Regularized models (L1/L2) or tree ensembles naturally partition redundant features (D17)."
            if n_clusters > 0
            else None
        ),
    )
    findings.append(f_clust)

    # D22 Overlap metrics support: inspect if present in near_duplicate_clusters or pairs
    has_cluster_overlap = any("min_pairwise_overlap" in c for c in clusters)
    if n_clusters == 0:
        f_overlap = Finding(
            id="REL_OVERLAP_SUPPORT",
            category="FEATURE_RELATIONSHIPS",
            status=STATUS_NOMINAL,
            statement="Zero near-duplicate feature clusters were identified; pairwise overlap screening not applicable.",
            source_section="relationships",
            source_keys=["near_duplicate_clusters"],
            evidence={"available": True, "n_clusters": 0},
        )
    elif has_cluster_overlap:
        min_overlaps = [c["min_pairwise_overlap"] for c in clusters if "min_pairwise_overlap" in c]
        overall_min = min(min_overlaps) if min_overlaps else 0
        f_overlap = Finding(
            id="REL_OVERLAP_SUPPORT",
            category="FEATURE_RELATIONSHIPS",
            status=STATUS_INFORMATIVE,
            statement=(
                f"Near-duplicate clusters are supported by empirical pairwise overlap, with a minimum of "
                f"{overall_min:,} shared observed rows backing any clustered pair."
            ),
            source_section="relationships",
            source_keys=["near_duplicate_clusters.min_pairwise_overlap"],
            evidence={"min_pairwise_overlap": overall_min, "available": True},
        )
    else:
        f_overlap = Finding(
            id="REL_OVERLAP_SUPPORT",
            category="FEATURE_RELATIONSHIPS",
            status=STATUS_UNAVAILABLE,
            statement="Pairwise overlap row counts backing clusters are unavailable in this artifact schema.",
            source_section="relationships",
            source_keys=["near_duplicate_clusters"],
            evidence={"available": False},
        )
    findings.append(f_overlap)

    # Histogram verification
    n_hist = len(hist_groups) if isinstance(hist_groups, dict) else len(hist_groups)
    f_hist = Finding(
        id="REL_HISTOGRAM_VERIFICATION",
        category="FEATURE_RELATIONSHIPS",
        status=STATUS_INFORMATIVE if n_hist > 0 else STATUS_NOMINAL,
        statement=(
            f"Verification of {n_hist} histogram groups confirms that row sums across bins "
            f"vary across instances rather than summing to constant row totals."
            if n_hist > 0
            else f"Verification identified 0 histogram groups requiring row sum checks."
        ),
        source_section="relationships",
        source_keys=["histogram_groups"],
        evidence={"n_histogram_groups": n_hist},
    )
    findings.append(f_hist)

    sec_status = _aggregate_status([f.status for f in findings])
    if n_90 > 0 or n_clusters > 0:
        headline = f"Extensive collinearity detected ({n_90} pairs with |rho| >= {0.90:.2f}, {n_clusters} near-duplicate clusters at |rho| >= {min_rho})."
    else:
        headline = f"Pairwise correlation screening across {n_scored:,} pairs identified {n_90} highly correlated pairs and {n_clusters} clusters."
    narrative = [f.statement for f in findings if f.status != STATUS_UNAVAILABLE]

    return SectionInterpretation(
        section="relationships",
        status=sec_status,
        policy="Descriptive only; redundant features are reported, never dropped automatically (D17).",
        headline=headline,
        findings=findings,
        narrative=narrative,
        caveats=["Correlations are evaluated over pairwise observed rows and do not imply causal relationships."],
    )


def interpret_signal(sig: Dict[str, Any]) -> SectionInterpretation:
    """Interpret univariate predictive signal and fold consistency."""
    if not isinstance(sig, dict):
        raise TypeError("signal must be a dictionary")
    required = ["n_columns_scored", "n_columns_not_scorable", "auc_band_counts", "top_columns"]
    for k in required:
        if k not in sig:
            raise KeyError(f"signal missing required key: {k}")

    n_scored = _require_non_negative_int(sig["n_columns_scored"], "signal.n_columns_scored")
    n_unscorable = _require_non_negative_int(sig["n_columns_not_scorable"], "signal.n_columns_not_scorable")
    bands = sig["auc_band_counts"]
    if not isinstance(bands, dict):
        raise TypeError("signal auc_band_counts must be a dictionary")
    top = list(sig["top_columns"])

    findings = []

    # Band distribution
    auc_threshold = 0.90
    n_ge_90 = int(bands.get("ge_0.90", 0))
    f_bands = Finding(
        id="SIG_AUC_BANDS",
        category="STATISTICAL_SIGNAL",
        status=STATUS_INFORMATIVE if n_ge_90 > 0 else STATUS_NOMINAL,
        statement=(
            f"Univariate ROC-AUC profiling scored {n_scored} features ({n_unscorable} unscorable due to zero variance); "
            f"{n_ge_90} features achieve oriented AUC >= {auc_threshold:.2f} on observed values."
        ),
        source_section="signal",
        source_keys=["n_columns_scored", "n_columns_not_scorable", "auc_band_counts"],
        evidence={
            "n_scored": n_scored,
            "n_unscorable": n_unscorable,
            "n_ge_0_90": n_ge_90,
            "auc_threshold": auc_threshold,
        },
        recommendation=(
            "Methodological guardrail: Univariate signal is computed on the full training set for exploratory "
            "characterization. Using these scores for global feature selection outside cross-validation folds "
            "would leak evaluation data and is strictly prohibited (D18)."
        ),
    )
    findings.append(f_bands)

    # Top feature stability
    if top:
        top_col = top[0]["column"]
        top_auc = _require_finite_float(top[0]["auc_oriented"], "top_columns[0].auc_oriented")
        top_dir = str(top[0]["direction"])
        top_stable = bool(top[0]["fold_sign_consistent"])
        f_top = Finding(
            id="SIG_TOP_FEATURES",
            category="STATISTICAL_SIGNAL",
            status=STATUS_INFORMATIVE,
            statement=(
                f"The highest individual oriented AUC is observed in {top_col} ({top_auc:.3f}, {top_dir}), "
                f"with fold-sign consistency confirmed across cross-validation folds ({top_stable})."
            ),
            source_section="signal",
            source_keys=["top_columns"],
            evidence={
                "top_column": top_col,
                "auc_oriented": top_auc,
                "direction": top_dir,
                "fold_sign_consistent": top_stable,
            },
        )
    else:
        f_top = Finding(
            id="SIG_TOP_FEATURES",
            category="STATISTICAL_SIGNAL",
            status=STATUS_UNAVAILABLE,
            statement="No ranked top features are available in the signal profile.",
            source_section="signal",
            source_keys=["top_columns"],
            evidence={"available": False},
        )
    findings.append(f_top)

    sec_status = _aggregate_status([f.status for f in findings])

    # Headline avoids claiming strong signal if n_ge_90 is 0
    if n_ge_90 > 0:
        headline = f"Strong univariate signal detected across {n_scored} features ({n_ge_90} features with oriented AUC >= {auc_threshold:.2f})."
    else:
        headline = f"Univariate ROC-AUC profiling scored {n_scored} features ({n_ge_90} features with oriented AUC >= {auc_threshold:.2f})."

    narrative = [f.statement for f in findings if f.status != STATUS_UNAVAILABLE]

    return SectionInterpretation(
        section="signal",
        status=sec_status,
        policy="Descriptive only; computed on the full training set and never used for feature selection (D18).",
        headline=headline,
        findings=findings,
        narrative=narrative,
        caveats=["Univariate AUC reflects individual monotonic separation on observed rows, not multivariate feature importance."],
    )


def interpret_leakage(leak: Dict[str, Any]) -> SectionInterpretation:
    """Interpret data leakage screens, row order dependency, and vehicle ID caveats."""
    if not isinstance(leak, dict):
        raise TypeError("leakage must be a dictionary")
    required = ["flags", "class_exclusive_values", "duplicate_rows", "row_order", "feature_drift", "thresholds"]
    for k in required:
        if k not in leak:
            raise KeyError(f"leakage missing required key: {k}")

    flags = leak["flags"]
    if not isinstance(flags, dict):
        raise TypeError("leakage 'flags' must be a dictionary")
    exclusive = leak["class_exclusive_values"]
    dups = leak["duplicate_rows"]
    if not isinstance(dups, dict):
        raise TypeError("leakage 'duplicate_rows' must be a dictionary")
    row_order = leak["row_order"]
    if not isinstance(row_order, dict):
        raise TypeError("leakage 'row_order' must be a dictionary")
    drift = leak["feature_drift"]
    if not isinstance(drift, dict):
        raise TypeError("leakage 'feature_drift' must be a dictionary")
    th = leak["thresholds"]
    if not isinstance(th, dict):
        raise TypeError("leakage 'thresholds' must be a dictionary")

    # Strict validation of nested row_order without silent defaults
    if "runs_test" not in row_order or not isinstance(row_order["runs_test"], dict):
        raise KeyError("leakage row_order missing required dict: runs_test")
    if "p_value" not in row_order["runs_test"]:
        raise KeyError("leakage row_order.runs_test missing required key: p_value")
    runs_p = _require_finite_float(row_order["runs_test"]["p_value"], "runs_test.p_value")

    if "blocks" not in row_order or not isinstance(row_order["blocks"], dict):
        raise KeyError("leakage row_order missing required dict: blocks")
    if "cramers_v" not in row_order["blocks"]:
        raise KeyError("leakage row_order.blocks missing required key: cramers_v")
    cramers_v = _require_finite_float(row_order["blocks"]["cramers_v"], "blocks.cramers_v")

    # Strict validation of feature_drift without silent defaults
    if "n_abs_rho_ge_threshold" not in drift:
        raise KeyError("leakage feature_drift missing required key: n_abs_rho_ge_threshold")
    n_drift = _require_non_negative_int(drift["n_abs_rho_ge_threshold"], "feature_drift.n_abs_rho_ge_threshold")
    if "max_abs_rho" not in drift:
        raise KeyError("leakage feature_drift missing required key: max_abs_rho")
    max_drift = _require_finite_float(drift["max_abs_rho"], "feature_drift.max_abs_rho")

    # Strict validation of duplicate_rows
    if "n_conflicting_label_groups" not in dups:
        raise KeyError("leakage duplicate_rows missing required key: n_conflicting_label_groups")
    n_conflicts = _require_non_negative_int(dups["n_conflicting_label_groups"], "duplicate_rows.n_conflicting_label_groups")

    # Strict validation of drift threshold from metadata
    if "drift_abs_rho" not in th:
        raise KeyError("leakage thresholds missing required key: drift_abs_rho")
    drift_th = _require_finite_float(th["drift_abs_rho"], "thresholds.drift_abs_rho")

    findings = []

    # Class-exclusive values
    n_exclusive = len(exclusive) if isinstance(exclusive, (dict, list)) else 0
    has_excl = bool(flags.get("class_exclusive_values_present", n_exclusive > 0))
    f_excl = Finding(
        id="LEAK_CLASS_EXCLUSIVE",
        category="LEAKAGE_GUARD",
        status=STATUS_ATTENTION if has_excl else STATUS_NOMINAL,
        statement=(
            f"Screening identified {n_exclusive} class-exclusive values acting as trivial shortcuts."
            if has_excl
            else "Zero class-exclusive feature values were identified; no sensor reading acts as a trivial classification shortcut."
        ),
        source_section="leakage",
        source_keys=["class_exclusive_values", "flags.class_exclusive_values_present"],
        evidence={"n_class_exclusive": n_exclusive, "flag": has_excl},
    )
    findings.append(f_excl)

    # Conflicting duplicate rows
    has_conflicts = bool(flags.get("duplicate_rows_with_conflicting_labels", n_conflicts > 0))
    f_conf = Finding(
        id="LEAK_CONFLICTING_DUPLICATES",
        category="LEAKAGE_GUARD",
        status=STATUS_ATTENTION if has_conflicts else STATUS_NOMINAL,
        statement=(
            f"Screening detected {n_conflicts} duplicate row groups with conflicting failure labels."
            if has_conflicts
            else "Zero duplicate row groups with conflicting class labels were found."
        ),
        source_section="leakage",
        source_keys=["duplicate_rows.n_conflicting_label_groups", "flags.duplicate_rows_with_conflicting_labels"],
        evidence={"n_conflicting_groups": n_conflicts, "flag": has_conflicts},
    )
    findings.append(f_conf)

    # Row order dependency
    order_flag = bool(flags.get("row_order_related_to_label", False))
    f_order = Finding(
        id="LEAK_ROW_ORDER",
        category="LEAKAGE_GUARD",
        status=STATUS_ATTENTION if order_flag else STATUS_NOMINAL,
        statement=(
            f"Row-order statistical tests show no evidence of label sorting or sequential batching "
            f"(runs test p = {runs_p:.4f}, block Cramer's V = {cramers_v:.4f})."
            if not order_flag
            else f"Row order exhibits potential label association (runs test p = {runs_p:.4f}, Cramer's V = {cramers_v:.4f})."
        ),
        source_section="leakage",
        source_keys=["row_order.runs_test.p_value", "row_order.blocks.cramers_v", "flags.row_order_related_to_label"],
        evidence={"runs_p": runs_p, "cramers_v": cramers_v, "flag": order_flag},
    )
    findings.append(f_order)

    # Feature drift with position
    drift_flag = bool(flags.get("features_drift_with_row_position", False))
    f_drift = Finding(
        id="LEAK_FEATURE_DRIFT",
        category="LEAKAGE_GUARD",
        status=STATUS_ATTENTION if drift_flag else STATUS_NOMINAL,
        statement=(
            f"Screening found zero features with meaningful position correlation ({n_drift} features with |rho| >= {drift_th:.2f}, max |rho| = {max_drift:.4f})."
            if not drift_flag
            else f"Position drift detected across {n_drift} features (max |rho| = {max_drift:.4f})."
        ),
        source_section="leakage",
        source_keys=["feature_drift.n_abs_rho_ge_threshold", "feature_drift.max_abs_rho", "flags.features_drift_with_row_position", "thresholds.drift_abs_rho"],
        evidence={"n_drifting": n_drift, "max_abs_rho": max_drift, "drift_threshold": drift_th, "flag": drift_flag},
    )
    findings.append(f_drift)

    sec_status = _aggregate_status([f.status for f in findings])
    any_flags = any(bool(v) for v in flags.values())
    headline = (
        "Leakage screening passed: zero class-exclusive values, zero duplicate conflicts, and no label ordering anomalies."
        if not any_flags
        else "Leakage screening raised review flags requiring scrutiny."
    )
    narrative = [f.statement for f in findings]

    return SectionInterpretation(
        section="leakage",
        status=sec_status,
        policy="Flags are review prompts, not verdicts (D19).",
        headline=headline,
        findings=findings,
        narrative=narrative,
        caveats=[
            "Critical limitation: Scania APS data lacks vehicle identifiers; the absence of leakage flags "
            "cannot rule out multi-snapshot dependency from identical trucks (D7, D19, OM6)."
        ],
    )


# --------------------------------------------------------------------------- #
# Master Aggregator
# --------------------------------------------------------------------------- #
def interpret_eda(results: Dict[str, Any]) -> Dict[str, Any]:
    """Pure entrypoint converting complete EDA results into structured interpretation."""
    if not isinstance(results, dict):
        raise TypeError("results must be a dictionary")
    if "metadata" not in results:
        raise KeyError("results missing 'metadata' section")
    if "sections" not in results:
        raise KeyError("results missing 'sections' dictionary")

    meta = results["metadata"]
    sections = results["sections"]

    required_sections = [
        "class_balance",
        "schema",
        "quality",
        "missingness",
        "distributions",
        "relationships",
        "signal",
        "leakage",
    ]
    for s in required_sections:
        if s not in sections:
            raise KeyError(f"Missing required EDA section: {s}")

    interp_meta = interpret_metadata(meta)
    interp_sections = {
        "class_balance": interpret_class_balance(sections["class_balance"]),
        "schema": interpret_schema(sections["schema"]),
        "quality": interpret_quality(sections["quality"]),
        "missingness": interpret_missingness(sections["missingness"]),
        "distributions": interpret_distributions(sections["distributions"]),
        "relationships": interpret_relationships(sections["relationships"]),
        "signal": interpret_signal(sections["signal"]),
        "leakage": interpret_leakage(sections["leakage"]),
    }

    # System-level status uses strict precedence: ATTENTION > INFORMATIVE > NOMINAL > UNAVAILABLE
    overall_status = _aggregate_status([s.status for s in interp_sections.values()])

    return {
        "metadata": interp_meta.to_dict(),
        "overall_status": overall_status,
        "sections": {k: v.to_dict() for k, v in interp_sections.items()},
    }
