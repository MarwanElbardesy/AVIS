"""EDA command-line runner (TRAINING data only).

    python -m src.eda.run_eda

Writes a deterministic ``eda_results.json`` (git-ignored) into reports/eda/.
Later checkpoints register more sections in SECTIONS; the runner, metadata and
JSON format stay the same.

The official test set is never opened here: this module imports only
``load_scania_train``.
"""

import argparse
from pathlib import Path

from src.data.scania import (
    PROJECT_ROOT,
    TRAIN_FILE,
    load_scania_train,
    split_features_target,
)
from src.eda import distributions, leakage, missingness, quality, relationships, schema, signal
from src.eda.utils import dumps_deterministic, library_versions, sha256_file

RESULTS_SCHEMA_VERSION = 1
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "eda"


def _section_class_balance(X, y, ctx):
    return quality.class_balance(y)


def _section_schema(X, y, ctx):
    return schema.describe_groups(list(X.columns))


def _section_quality(X, y, ctx):
    return quality.quality_report(X, y)


def _section_missingness(X, y, ctx):
    return missingness.missingness_report(X, y, seed=ctx["seed"])


def _section_distributions(X, y, ctx):
    return distributions.distribution_report(X, y)


def _section_relationships(X, y, ctx):
    return relationships.relationships_report(X)


def _signal_result(X, y, ctx):
    """Compute the univariate-signal report once per run and share it."""

    if "signal" not in ctx:
        ctx["signal"] = signal.signal_report(X, y, seed=ctx["seed"])
    return ctx["signal"]


def _section_signal(X, y, ctx):
    return _signal_result(X, y, ctx)


def _section_leakage(X, y, ctx):
    return leakage.leakage_report(X, y, _signal_result(X, y, ctx)["columns"])


# name -> function(X, y, ctx) returning a JSON-friendly object. Order = run order.
SECTIONS = {
    "class_balance": _section_class_balance,
    "schema": _section_schema,
    "quality": _section_quality,
    "missingness": _section_missingness,
    "distributions": _section_distributions,
    "relationships": _section_relationships,
    "signal": _section_signal,
    "leakage": _section_leakage,
}


def run_eda(train_path=None, out_dir=None, seed=42, sections=None):
    """Run the EDA sections on the training set; return the results dict and
    write it to ``<out_dir>/eda_results.json``."""

    train_path = Path(train_path or TRAIN_FILE)
    out_dir = Path(out_dir or DEFAULT_OUT_DIR)

    train = load_scania_train(train_path)
    X, y = split_features_target(train)

    selected = list(SECTIONS) if sections is None else list(sections)
    unknown = [name for name in selected if name not in SECTIONS]
    if unknown:
        raise ValueError(f"Unknown EDA section(s): {unknown}. Available: {list(SECTIONS)}")

    ctx = {"seed": seed}
    results = {
        "metadata": {
            "results_schema_version": RESULTS_SCHEMA_VERSION,
            "dataset": "Scania APS Failure - TRAINING set only",
            "train_file_name": train_path.name,
            "train_file_sha256": sha256_file(train_path),
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "seed": seed,
            "libraries": library_versions(),
            "sections": selected,
        },
        "sections": {name: SECTIONS[name](X, y, ctx) for name in selected},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eda_results.json").write_text(dumps_deterministic(results), encoding="utf-8", newline="\n")
    return results


def format_summary(results):
    """Short console summary of whatever sections were run."""

    lines = []
    meta = results["metadata"]
    lines.append(f"EDA on {meta['train_file_name']}: {meta['rows']} rows x {meta['features']} features")
    lines.append(f"sha256: {meta['train_file_sha256']}")

    balance = results["sections"].get("class_balance")
    if balance:
        low, high = balance["positive_rate_ci95"]
        lines.append(
            f"classes: {balance['negatives']} neg / {balance['positives']} pos "
            f"(positive rate {balance['positive_rate']:.4%}, 95% CI {low:.4%}-{high:.4%})"
        )

    grp = results["sections"].get("schema")
    if grp:
        lines.append(
            f"schema: {grp['n_groups']} prefix groups; histogram candidates: "
            f"{', '.join(grp['histogram_candidate_prefixes']) or 'none'}; "
            f"unparsed names: {len(grp['unparsed_columns'])}"
        )

    qual = results["sections"].get("quality")
    if qual:
        lines.append(
            f"quality: {qual['missing_cells']} missing cells in {qual['features_with_missing']} features; "
            f"{qual['duplicate_rows']} duplicate rows"
        )
        lines.append(f"  constant: {qual['constant_columns'] or 'none'}")
        lines.append(f"  near-constant (>=99.9% one value): {len(qual['near_constant_columns'])}")
        lines.append(f"  all-missing: {qual['all_missing_columns'] or 'none'}")
        lines.append(f"  duplicate column groups: {qual['duplicate_column_groups'] or 'none'}")
        lines.append(f"  columns with negative values: {len(qual['columns_with_negative_values'])}")
        lines.append(f"  non-integer columns: {len(qual['non_integer_columns'])}")
        lines.append(f"  saturation candidates: {len(qual['saturation_candidates'])}")
        lines.append(f"  non-numeric columns: {qual['non_numeric_columns'] or 'none'}")

    miss = results["sections"].get("missingness")
    if miss:
        ta = miss["target_association"]
        rows = miss["rows"]
        lines.append(
            f"missingness: {miss['n_columns_with_missing']}/{miss['n_columns_total']} columns have missing values; "
            f"buckets {miss['bucket_counts']}"
        )
        lines.append(
            f"  target association (indicator AUC, Fisher + BH FDR @ {ta['alpha']}): "
            f"{ta['n_significant_fdr']}/{ta['n_tested']} significant, "
            f"{ta['n_notable_effect']} with |AUC-0.5|>={ta['notable_auc_delta']} "
            f"({ta['n_higher_missing_in_positives']} higher missing in failures, "
            f"{ta['n_lower_missing_in_positives']} lower)"
        )
        if "n_significant_higher_missing_in_positives" in ta:  # absent in results written before 2d-1
            lines.append(
                f"  significant: {ta['n_significant_higher_missing_in_positives']} higher / "
                f"{ta['n_significant_lower_missing_in_positives']} lower missing in failures; "
                f"notable: {ta['n_notable_higher_missing_in_positives']} higher / "
                f"{ta['n_notable_lower_missing_in_positives']} lower; "
                f"significant AND notable: {ta['n_significant_and_notable']}"
            )
        lines.append(f"  strongest columns: {', '.join(ta['strongest_columns'][:5]) or 'none'}")
        lines.append(
            f"  missing values per row: mean {rows['negative']['mean']:.1f} (neg) vs "
            f"{rows['positive']['mean']:.1f} (pos); AUC of row missing-count = "
            f"{rows['auc_missing_count_vs_target']:.3f}"
        )
        lines.append(
            f"  co-missing clusters (phi>={miss['comissing_cluster_min_phi']}): "
            f"{len(miss['comissing_clusters'])}; identical-pattern groups: "
            f"{len(miss['identical_pattern_groups'])}"
        )
        pats = miss["row_patterns"]
        lines.append(
            f"  distinct row patterns: {pats['n_distinct_patterns']} "
            f"(top {len(pats['top_patterns'])} cover {pats['top_k_coverage']:.1%} of rows)"
        )

    dist = results["sections"].get("distributions")
    if dist:
        lines.append(
            f"distributions: {dist['n_columns_profiled']} columns profiled; "
            f"{dist['n_heavy_tailed']} heavy-tailed (|skew|>={dist['thresholds']['heavy_skew']:g}), "
            f"{dist['n_zero_inflated']} zero-inflated (>={dist['thresholds']['zero_inflated']:.0%} zeros), "
            f"{dist['n_iqr_zero']} with IQR=0"
        )

    rel = results["sections"].get("relationships")
    if rel:
        lines.append(
            f"relationships: {rel['n_pairs_scored']} pairs scored; |rho|>=0.95: {rel['n_pairs_abs_rho_ge_0_95']}; "
            f"near-duplicate clusters: {len(rel['near_duplicate_clusters'])}"
        )
        for prefix, h in rel["histogram_groups"].items():
            total = (
                "constant total" if h.get("total_is_constant")
                else f"total CV {h['row_total_cv']:.3f}" if h.get("row_total_cv") is not None else "no complete rows"
            )
            lines.append(
                f"  histogram {prefix}: {h['n_complete_rows']} complete rows, {total}, "
                f"adjacent-bin rho {h['mean_adjacent_bin_rho']:.2f}"
            )

    sig = results["sections"].get("signal")
    if sig:
        lines.append(
            f"signal (descriptive only; oriented AUC on observed rows): bands {sig['auc_band_counts']}; "
            f"{sig['n_low_support']} low-support columns"
        )
        top = ", ".join(f"{t['column']} {t['auc_oriented']:.3f}" for t in sig["top_columns"][:5])
        lines.append(f"  top: {top}")

    leak = results["sections"].get("leakage")
    if leak:
        raised = [name for name, on in leak["flags"].items() if on]
        blocks = leak["row_order"]["blocks"]
        lines.append(
            f"leakage/ordering: flags raised: {', '.join(raised) if raised else 'none'}; "
            f"block p={blocks['p_value']:.3g} (V={blocks['cramers_v']:.3f}), "
            f"runs-test p={leak['row_order']['runs_test']['p_value']:.3g}, "
            f"max |feature-vs-row-position rho|={leak['feature_drift']['max_abs_rho']:.3f}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sections", nargs="*", default=None, help=f"subset of {list(SECTIONS)}")
    args = parser.parse_args()

    results = run_eda(args.train_path, args.out_dir, args.seed, args.sections)
    print(format_summary(results))
    print(f"\nSaved: {Path(args.out_dir or DEFAULT_OUT_DIR) / 'eda_results.json'}")


if __name__ == "__main__":
    main()
