# Architecture Decisions

## D1 - Training and test loading are separate
`load_scania_train()` never touches the test file; `load_scania_test()` is reserved for final
evaluation. `load_scania_data()` is kept for backward compatibility. A test enforces that train
loading works with the test file absent.

## D2 - Header detection instead of `skiprows=20`
The loader finds the line starting with `class,`; 20 remains only as a fallback.

## D3 - Missingness is signal, so it is preserved
Missingness is strongly associated with failure. Linear/RF pipelines use median imputation with
`add_indicator=True`; gradient boosting keeps NaN natively. Never impute-and-forget.

## D4 - Constant columns dropped inside the pipeline
`cd_000` (and any constant column) is dropped by a transformer fitted per fold, not by a manual
global step.

## D5 - Cost-aware evaluation
Primary metric is the APS cost (10 FP + 500 FN). PR-AUC reported alongside ROC-AUC because
positives are ~1.7%. Threshold search is vectorised and tested against brute force.

## D6 - Honest threshold estimate
A threshold tuned and scored on the same OOF rows is optimistic. Reports include both that number
(labelled OPTIMISTIC) and a cross-fitted cost.

## D7 - Plain stratified K-fold
APS rows are independent snapshots with no vehicle id, so grouped splitting is not possible or
needed here. Revisit for any dataset with several rows per vehicle.

## D8 - Baselines
Logistic regression (scaled), random forest, histogram gradient boosting. Hyperparameters are
untuned defaults; tuning is a later milestone and must happen inside CV.

## D9 - EDA is a separate, read-only package (`src/eda/`)
EDA code lives beside, not inside, the Milestone 1 modules. It reuses `load_scania_train`,
`split_features_target` and `constant_columns` unchanged and writes nothing into the pipeline.
EDA findings never modify preprocessing automatically; any change is a deliberate, documented
decision made later.

## D10 - EDA uses the training set only (enforced)
No EDA module may reference the test loader or path. A static test scans `src/eda/*.py` for
`load_scania_test`, `load_scania_data`, `TEST_FILE` and the test file name, and a runtime test runs
the whole EDA with the test path pointing at a non-existent file.

## D11 - Sparse columns are never flagged for removal
Quality checks flag only constant, all-missing, duplicate, saturated or non-numeric columns.
A high missing rate is reported as a fact, not a defect, because missingness is informative here (D3).

## D12 - Reproducible, deterministic results artifact
`eda_results.json` carries the input file SHA-256, row/feature counts, seed and library versions,
has sorted keys, strict JSON (NaN -> null), LF newlines and no timestamps, so identical input gives
byte-identical output (tested). The JSON is git-ignored; the Markdown report and figures are committed.

## D13 - Structural-only schema inference
Feature groups come from names (`prefix_NNN`) alone. Histogram groups are reported as *candidates*
(10 contiguous bins from 0) and verified against values in checkpoint 2c.

## D14 - Missingness is analysed, never acted on
`src/eda/missingness.py` is descriptive. Its report carries an explicit policy string and a test
asserts that no "drop/remove" keys or columns exist. No column is excluded from any table, including
columns that are 90%+ missing. Using any ranking for selection requires re-deriving it inside CV folds.

## D15 - Effect sizes next to p-values
With 60,000 rows nearly every association is "significant". Each column therefore reports the
missing-indicator AUC (= 0.5 + (P(miss|pos) - P(miss|neg))/2), lift and odds ratio alongside the
Fisher exact p-value and its Benjamini-Hochberg q-value (FDR across all tested columns). A
"notable effect" is |AUC - 0.5| >= 0.05. Odds ratios use a +0.5 correction only when a cell is zero.

## D16 - Co-missingness via phi + average-linkage clustering
Missing indicators are compared with the phi coefficient; columns are clustered with average linkage
at phi >= 0.9 (positive association only: "go missing together"). Exact-identical masks are reported
separately. Cluster output is sorted and deterministic. Association stability is re-measured on
stratified validation folds (min/max indicator AUC per column); this is a diagnostic, not a selector.

## D17 - Rank correlation by matrix algebra, tolerant of missing values
`pairwise_rank_correlation` ranks each column once over its observed values, centres the ranks, and
computes Pearson over rows where both columns are observed using matrix products (no per-pair loop;
~14k pairs in well under a second). It is exactly Spearman when nothing is missing and agrees with
pandas' pairwise Spearman to ~1e-4 under informative missingness (tested). Pairs sharing fewer than
100 rows, and constant columns, are NaN. A mutation test showed the overlap-mean term matters only
when missingness is informative, so a dedicated test covers that case.

## D18 - Univariate signal is descriptive only
Per-feature AUC is computed on observed rows (Mann-Whitney via average ranks, equal to sklearn's
`roc_auc_score`), with a Hanley-McNeil standard error (checked by Monte Carlo), per-fold min/max,
fold sign consistency and a low-support flag (< 30 observed failures). The missing-indicator AUC is
shown beside it. Because it is computed on the whole training set, using it to select features would
leak; any selection must be re-derived inside CV folds. The mutual-information step from the original
plan was dropped: it needs imputation and is stochastic, and AUC covers the same monotone effects.

## D19 - Leakage checks are review prompts, not verdicts
Checks: label-vs-row-order (block positive rates with chi-square + Cramer's V, Wald-Wolfowitz runs
test), feature-vs-row-position drift, near-perfect single features (value AUC >= 0.98, missing-
indicator AUC >= 0.90), values seen in only one class (both directions), duplicate rows with
conflicting labels. Thresholds are constants in `leakage.py`. APS has no vehicle id, so same-truck
near-duplicates cannot be detected; this limitation is written into the report policy.

## D20 - Repo-wide test-set guard
`tests/test_test_set_guard.py` scans every module under `src/` (except the loader definition in
`src/data/scania.py`) for `load_scania_test`, `load_scania_data`, `TEST_FILE` and the test file name.
This makes the README claim enforceable beyond `src/eda`. The future final-evaluation module will be
a deliberate, reviewed exception.

## D21 - Distribution profile never transforms anything
Heavy-tailed (|skew| >= 5) and zero-inflated (>= 50% zeros) flags only describe. Tukey-fence outlier
shares are reported as NaN when the IQR is zero (the fences collapse onto the median and "outlier"
would just mean "not the median").

## D22 - Additive output fields for the report (2d-1)
Shared-row counts and direction splits were added so the report can state how much data stands behind
each correlation and which way significant missingness points. Nothing existing changed:
* `relationships.pairwise_overlap_counts(X)` is new. `top_correlated_pairs(..., overlap=None)` and
  `near_duplicate_clusters(..., overlap=None)` return exactly the old output when `overlap` is omitted;
  with it, pairs gain `n_overlap` and clusters gain `min_pairwise_overlap`. `relationships_report`
  passes it, so the JSON gains those fields.
* `target_association` gains five integers: significant higher/lower, notable higher/lower, and
  significant-and-notable. The existing `n_higher/lower_missing_in_positives` keep their meaning (all
  tested columns).
* `results_schema_version` stays 1 because the change is purely additive; consumers must tolerate absent
  keys in results written earlier (the console summary does, tested).
* Mutation checks found that a first fixture with symmetric counts could not detect a swapped direction
  or an and/or error; a fixture with deliberately distinct counts (5/2/3/1/4) now guards both.

## Open methodology issues (found on review; none fixed yet, Milestone 1 untouched)
| # | Issue | Owner |
|---|-------|-------|
| OM1 | `cross_fitted_cost` is a raw sum over 60,000 rows but the test set has 16,000; final comparison must use cost per row (or scaled cost). | Milestone 3 |
| OM2 | Models are compared on one 5-fold split with ~1,000 positives (500 per miss); rankings can be noise. Needs repeated stratified CV with spread and paired comparison before choosing a model. | Milestone 3 |
| OM3 | Prior shift: test has ~2.3% positives vs ~1.7% in train. A threshold from train out-of-fold scores may not transfer, and fold models (80% of data) score differently from a final model fit on 100%. The final-evaluation protocol must be fixed before the test set is looked at. | Milestone 3 |
| OM4 | Cross-fitted thresholds use OOF scores produced by five different fold models; scales may differ slightly. | Milestone 3 |
| OM5 | `at_cost_optimal_threshold` precision/recall are tuned and scored on the same OOF rows (optimistic); only the key name says so. | Milestone 3 |
| OM6 | No vehicle id: same-truck near-duplicate snapshots could make CV optimistic. Disclose; the official test set is the only real check. | Disclosure |
