# AVIS v1.0 — Handoff
> **Advanced Vehicle Intelligence System** *(formerly CVI — Commercial Vehicle Intelligence)*

Last handoff: 2026-11-16

## Project goal

Build AVIS (Advanced Vehicle Intelligence System, formerly CVI — Commercial Vehicle Intelligence) v1.0: a reliable, portfolio-ready predictive-maintenance system using the Scania APS failure-prediction dataset.

The repository is the source of truth. Do not rely on prior AI chat history when the project files can establish the current state.

## Working rules

- Inspect the repository before making changes.
- Continue from the existing state; do not repeat or rebuild completed work.
- Preserve all existing files and uncommitted changes.
- Never overwrite files blindly.
- Do not modify Milestone 1 code unless explicitly approved.
- Keep development and model selection strictly training-data-only.
- Do not open, inspect, or use the official test dataset until the project’s final evaluation stage is explicitly approved.
- Keep scope limited to AVIS v1.0 (formerly CVI v1.0). Defer research extensions and v2.0 ideas.
- Run relevant tests after changes and report exact results.
- Update `PROGRESS.md` and `DECISIONS.md` when project status or technical decisions change.
- Before finishing a work session, record what changed, tests run, unresolved issues, and the exact next action in this file.

## Known project context (verify against repository)

- Milestone 1: previously reported as completed; verify against actual files and tests.
- Milestone 2a: reported completed.
- Milestone 2b: reported completed; 122 tests were previously reported passing.
- Milestone 2c: was being implemented when the previous Claude session hit its usage limit. Its actual completion state is UNKNOWN and must be established by inspecting the working tree.
- Milestone 2d: not started according to the latest handoff report.
- Baseline cross-validation: execution status is UNKNOWN. A previous review found no `reports/cv_results.json` in the reviewing copy, but that does not establish the state of the user's local machine. Check the actual repository and ask before rerunning a long job.

## Approved Milestone 2c scope

The previous session was implementing the following additive, training-data-only analyses:

1. Feature distribution summaries: robust quantiles, skew, kurtosis, zero-inflation, IQR outlier fractions, and tail ratios.
2. Relationships: rank-correlation matrix, near-duplicate feature clusters, and verification of seven histogram candidates.
3. Univariate signal: per-feature AUC on observed rows, Hanley–McNeil standard error, per-fold stability, and low-support flags. Descriptive only; not for feature selection.
4. Leakage and ordering checks: label association with row order, block rates, runs test, feature drift by row index, near-separating features, and values appearing in only one class.
5. A repository-wide static test that development modules under `src/` do not reference/open the official test set, with any narrowly necessary loader-definition exception documented.
6. An “Open methodology issues” section in `DECISIONS.md`.

The previous session stated that this work was in progress, not completed. Inspect actual files, diffs, and tests to determine what was already implemented. Do not blindly rerun or recreate it.

## Known methodology issues to track

These were identified in a prior code review and are OPEN until verified/resolved/documented:

- Raw total APS costs are not directly comparable across datasets of different sizes (training vs. official test). Report cost per row or another explicitly normalized measure alongside totals.
- A single 5-fold split may produce noisy model rankings. Consider repeated CV and report variability before selecting a model, if within v1.0 scope and compute budget.
- The observed positive prevalence differs between train (~1.67%) and test (~2.34%). Treat this as prior/prevalence shift. Define the final threshold-transfer and evaluation protocol before accessing the official test set.
- The README's test-set isolation claim may be broader than existing automated enforcement. Verify the static test covers all relevant development code.
- Precision/recall at a threshold optimized on the same OOF rows are optimistic and should be labelled accordingly.
- The APS data has no vehicle ID. Potential repeated/near-duplicate truck snapshots cannot be grouped; disclose this limitation rather than claiming vehicle-independent validation.
- Earlier review noted that cross-fitted threshold selection from other folds' OOF scores is not fully independent because models producing those scores may have trained on the held-out fold. Do not describe this estimate as fully independent. Decide whether nested CV is necessary for the v1.0 claims.

## Immediate takeover procedure

Before editing anything:

1. Read `PROGRESS.md`, `DECISIONS.md`, and `README.md`.
2. Inspect the complete project tree and `git status`.
3. Review uncommitted diffs; do not discard or reset anything.
4. Identify files created/modified for Milestone 2c and inspect their implementation.
5. Run only fast, relevant tests first; do not launch long CV/model training without approval.
6. Report:
   - what is complete;
   - what is partially implemented;
   - what remains;
   - uncommitted changes;
   - test results actually observed;
   - the single next action.

Wait for approval before making changes.

## Session log

### 2026-11-16 — Account handoff

- Previous Claude account hit its usage limit during Milestone 2c.
- No reliable final completion report was available.
- New session must inspect repository state before continuing.
