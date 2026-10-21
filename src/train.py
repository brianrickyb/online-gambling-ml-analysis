"""End-to-end training: load -> clean -> compare models -> (tune) -> evaluate -> save.

Usage:
    python -m src.train            # compare all models, keep the best
    python -m src.train --tune     # also grid-search hyperparameters (~3 min)
    python -m src.train --fast     # skip random forest (slowest) during comparison
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn

from src.config import (
    DATA_PATH, FIGURE_DIR, LABEL_COL, MODEL_INFO_PATH, MODEL_PATH, REPORT_DIR, TEXT_COL,
    THRESHOLD,
)
from src.evaluation import (
    error_analysis, evaluate_model, plot_confusion_matrix, plot_model_comparison,
    plot_precision_recall, top_features,
)
from src.model import (
    build_pipeline, compare_models, get_candidate_models, select_best_model, split_data,
    tune_model,
)
from src.preprocessing import clean_data, load_data


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(fast=False, tune=False):
    sys.stdout.reconfigure(encoding="utf-8")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("1. Loading & cleaning data")
    raw = load_data(DATA_PATH)
    df = clean_data(raw)
    print(f"   {len(raw)} rows -> {len(df)} after dropping duplicates")
    print(f"   label balance: {df[LABEL_COL].value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = split_data(df[TEXT_COL], df[LABEL_COL])
    print(f"   train={len(X_train)}  test={len(X_test)}")

    print("2. Comparing models with 5-fold cross-validation (train set only)")
    models = get_candidate_models()
    if fast:
        models.pop("random_forest")
    cv_results = compare_models(X_train, y_train, models)
    best_name = select_best_model(cv_results)
    print(f"   chosen (one-standard-error rule): {best_name}")

    best_params = None
    if tune and best_name == "logistic_regression":
        print("3. Tuning hyperparameters with GridSearchCV (train set only)")
        search = tune_model(X_train, y_train)
        model = search.best_estimator_
        best_params = {k: list(v) if isinstance(v, tuple) else v for k, v in search.best_params_.items()}
        print(f"   best params: {best_params}  CV F1={search.best_score_:.4f}")
    else:
        if tune:
            print(f"   (tuning grid is defined for logistic_regression only, skipping for {best_name})")
        print("3. Training chosen model on the full train set")
        model = build_pipeline(models[best_name])
        model.fit(X_train, y_train)

    print("4. Evaluating on the held-out test set")
    metrics = evaluate_model(model, X_test, y_test)
    print(metrics["report"])
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")

    plot_model_comparison(cv_results, FIGURE_DIR / "model_comparison.png")
    plot_confusion_matrix(model, X_test, y_test, FIGURE_DIR / "confusion_matrix.png")
    plot_precision_recall(model, X_test, y_test, FIGURE_DIR / "precision_recall.png")

    errors = error_analysis(model, X_test, y_test)
    errors.to_csv(REPORT_DIR / "errors.csv", index=False, encoding="utf-8")
    print(f"   {len(errors)} misclassified comments -> reports/errors.csv")

    features = top_features(model)
    if features:
        print("   top judol features:", [name for name, _ in features["judol"][:10]])

    test_metrics = {k: v for k, v in metrics.items() if k != "report"}
    report = {
        "best_model": best_name,
        "best_params": best_params,
        "n_rows_raw": len(raw),
        "n_rows_clean": len(df),
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "top_features": features,
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Everything needed to reproduce, audit, and monitor this exact model.
    test_scores = model.predict_proba(X_test)[:, 1]
    trained_at = datetime.now(timezone.utc)
    model_info = {
        "version": trained_at.strftime("%Y%m%d-%H%M%S"),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "model": best_name,
        "params": best_params,
        "threshold": THRESHOLD,
        "data_sha256": file_sha256(DATA_PATH),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_metrics": test_metrics,
        # Baseline for monitoring: what "normal traffic" looked like at training time.
        "baseline": {
            "judol_rate": float((test_scores >= THRESHOLD).mean()),
            "score_quantiles": {
                str(q): float(np.quantile(test_scores, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
        },
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    MODEL_INFO_PATH.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    joblib.dump(model, MODEL_PATH)
    print(f"5. Saved model v{model_info['version']} -> models/{MODEL_PATH.name}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="skip random forest")
    parser.add_argument("--tune", action="store_true", help="grid-search hyperparameters")
    args = parser.parse_args()
    main(fast=args.fast, tune=args.tune)
