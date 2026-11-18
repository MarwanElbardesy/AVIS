# AVIS Progress
> **Advanced Vehicle Intelligence System** *(formerly CVI — Commercial Vehicle Intelligence)*

## Status
* **Current State**: Milestone 2d-2 Complete (2026-11-17).
* **Test Suite**: 205 tests passing (100% pass rate across synthetic unit test suite).
* **Next Action**: Milestone 2d-3 (figure generation from training CSV + JSON).

## Done
- Project structure, Python 3.12 venv, core libraries
- Milestone 1: leakage-safe ML foundation
  - Robust loader, train/test separation, data audit (constant columns, missingness vs target)
  - Per-fold preprocessing, APS cost metric + threshold search
  - OOF cross-validation with cross-fitted threshold, 3 baseline models, CLI runner
  - 46 automated tests (including leakage canary tests)

- Milestone 2 (EDA)
  - [x] 2a: feature-group schema, data-quality checks, class balance, CLI + results JSON (40 tests)
  - [x] 2b: missingness analysis (per-column/row, target association + FDR, co-missing clusters, group structure, row patterns, fold stability)
  - [x] 2c: distributions, relationships (rank correlation, near-duplicate clusters, histogram verification), univariate signal (descriptive), leakage/ordering checks, repo-wide test-set guard
  - [ ] 2d: Markdown report + figures
    - [x] 2d-1: additive fields (pair/cluster shared-row counts; significant/notable direction splits) (192 tests total)
    - [x] 2d-2: rule-based interpretation text layer (numbers templated from the JSON) (205 tests total)
    - [ ] 2d-3: figures (training CSV + JSON; test set never opened) [PENDING]
    - [ ] 2d-4: `build_report` CLI, determinism tests, docs [PENDING]

## Next
0. Milestone 2d: Markdown report + figures from `eda_results.json`.
1. Run `python -m src.models.train_baselines` on the real training data and review results (status unconfirmed: results not yet seen). Resolve OM1-OM5 in DECISIONS.md before model selection.
2. Hyperparameter tuning inside CV (cost-aware), pick one model.
3. One-time final evaluation on the official test set.
4. SHAP explainability + sensor-level failure analysis.
5. Health scoring and maintenance recommendations.
6. FastAPI service, then dashboard.
7. VED / anomaly detection (original plan; revisit scope).
