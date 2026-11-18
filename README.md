# AVIS — Advanced Vehicle Intelligence System

> **High-Reliability Predictive Maintenance & Diagnostics for Heavy Commercial Vehicles**<br>
> *(Formerly CVI — Commercial Vehicle Intelligence)*

AVIS (Advanced Vehicle Intelligence System) is a predictive-maintenance and diagnostic intelligence platform engineered for heavy commercial transport fleets. Built on the Scania Air Pressure System (APS) dataset, AVIS is designed to minimize catastrophic on-road breakdowns through cost-sensitive machine learning, deterministic exploratory data analysis (EDA), and strict test-set isolation.

---

## Technical Overview

Heavy commercial fleets face severe asymmetric breakdown costs: an unplanned on-road catastrophic failure costs substantially more in towing, operational disruption, and emergency repairs than proactive workshop servicing. AVIS addresses this through:
- **Asymmetric Cost Optimization**: Direct optimization against the industrial domain cost matrix:
  $$\text{Cost} = 10 \times \text{False Positives} + 500 \times \text{False Negatives}$$
- **Strict Leakage-Safe Architecture**: Enforced separation between training and test distributions. Official test sets are permanently guarded from exploratory and model training code via automated AST testing.
- **Deterministic EDA & Rule-Based Interpretation**: A test-backed diagnostic pipeline that transforms raw multi-sensor telemetry into structured, evidence-backed findings (`NOMINAL`, `INFORMATIVE`, `ATTENTION`, `UNAVAILABLE`) with zero hallucinated figures and strict epistemic guardrails.
- **Cross-Fitted Threshold Estimation**: Decision thresholds are tuned strictly on out-of-fold validation scores from complementary folds, mitigating in-sample optimization bias.

---

## Status & Milestones

- **Current State**: **Milestone 2d-2 Complete** (Nov 2026).
- **Test Suite**: **205 automated unit tests passing (100% pass rate across synthetic unit test suite)**.
- **Implemented Milestones**:
  - `Milestone 1`: Leakage-safe ML foundation, robust data loader, per-fold median imputer, APS cost metric, 5-fold OOF cross-validation, baseline classifiers (46 tests).
  - `Milestone 2a–2c`: Multi-sensor EDA suite covering schema family grouping (170 features across 74 families), data quality screening, phi missingness clustering, Spearman rank correlation, univariate signal screening, and index-order leakage checks (133 tests).
  - `Milestone 2d-1`: Additive report fields for pairwise observation overlaps and directional missingness splits (8 tests).
  - `Milestone 2d-2`: Rule-based interpretation layer (`src/eda/interpret.py`) with strict evidence traceability and anti-hallucination verification (13 tests).
- **Planned Next Steps (tracked in PROGRESS.md)**: Milestone 2d-3 (figure generation from training set + JSON) and Milestone 2d-4 (automated report generation CLI).

No model results are published in this repository until they are produced by running the code on the real dataset.

See `PROGRESS.md` and `DECISIONS.md` for complete milestone tracking and architectural decisions (D1–D23).

---

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the official dataset files in `data/raw/scania/`:
- `aps_failure_training_set.csv`
- `aps_failure_test_set.csv`
- `aps_failure_description.txt`

*(Raw datasets are never committed to version control).*

---

## Verification & Execution

```powershell
# Run the complete automated test suite (205 tests)
python -m pytest

# Run 5-fold cross-validation baselines (TRAINING set only)
python -m src.models.train_baselines

# Run the complete multi-sensor EDA pipeline (TRAINING set only)
python -m src.eda.run_eda
```

---

## Evaluation & Operational Protocol

1. **Industrial Metric**: Evaluated strictly on the Scania APS cost function ($C = 10 \cdot \text{FP} + 500 \cdot \text{FN}$).
2. **Per-Fold Isolation**: All transformations, scalers, and imputers are fitted strictly inside each training fold to eliminate validation leakages.
3. **Cross-Fitted Thresholding**: Operational decision thresholds are estimated across out-of-fold validation distributions (mitigating optimistic in-sample threshold bias, while recognizing inter-fold model dependencies as noted in OM7).
4. **Test-Set Guard**: The official test set (`aps_failure_test_set.csv`) is permanently restricted to final evaluation (`load_scania_test()`), verified continuously via repository-wide automated AST guards (`test_test_set_guard.py`).
5. **Domain Limitations**: The APS dataset does not contain vehicle identifiers (as documented in D19 and OM6); evaluation protocol treats snapshots as independent observations while disclosing this limitation.
