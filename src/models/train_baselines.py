"""Cross-validate the baseline models on the TRAINING set only.

Run from the project root:
    python -m src.models.train_baselines
The official test file is never opened by this script.
"""

import argparse
import json
from pathlib import Path

from src.data.audit import audit_training_data
from src.data.scania import PROJECT_ROOT, load_scania_train, split_features_target
from src.evaluation.cv import evaluate_pipeline_cv
from src.models.baseline import make_baseline_models


def run(train_path=None, n_splits=5, seed=42, fast=False, only=None, out_path=None):
    train = load_scania_train(train_path)
    X, y = split_features_target(train)

    results = {"data_audit": audit_training_data(X, y), "models": {}}

    for name, pipeline in make_baseline_models(seed=seed, fast=fast).items():
        if only and name not in only:
            continue
        print(f"Cross-validating {name} ({n_splits}-fold)...", flush=True)
        results["models"][name] = evaluate_pipeline_cv(pipeline, X, y, n_splits=n_splits, seed=seed)

    out_path = Path(out_path or PROJECT_ROOT / "reports" / "cv_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'model':<26}{'cost (honest)':>14}{'PR-AUC':>10}{'ROC-AUC':>10}{'recall':>9}{'prec':>8}")
    for name, res in results["models"].items():
        m = res["at_cost_optimal_threshold"]
        print(
            f"{name:<26}{res['cost_cross_fitted_threshold']:>14}"
            f"{m['pr_auc']:>10.4f}{m['roc_auc']:>10.4f}{m['recall']:>9.3f}{m['precision']:>8.3f}"
        )
    print(f"\nSaved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="small models for a smoke test")
    parser.add_argument("--only", nargs="*", default=None, help="subset of model names")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run(args.train_path, args.splits, args.seed, args.fast, args.only, args.out)


if __name__ == "__main__":
    main()
