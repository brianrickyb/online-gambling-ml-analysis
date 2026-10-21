import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import LABEL_NAMES


def _scores(model, X):
    """Continuous score for ROC-AUC: probability if available, else margin"""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate_model(model, X_test, y_test):
    """Evaluate trained model; returns a dict of metrics and the text report"""
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, _scores(model, X_test)),
        "report": classification_report(
            y_test, predictions, target_names=list(LABEL_NAMES.values()), digits=4,
        ),
    }


def plot_confusion_matrix(model, X_test, y_test, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=list(LABEL_NAMES.values()), cmap="Blues", ax=ax,
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_precision_recall(model, X_test, y_test, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_test, _scores(model, X_test), ax=ax, name="model")
    ax.set_title("Precision-Recall curve (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_model_comparison(cv_results, path):
    df = pd.DataFrame(cv_results).T[["accuracy", "precision", "recall", "f1"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot.bar(ax=ax, rot=0)
    ax.set_ylim(max(0.0, df.values.min() - 0.05), 1.0)
    ax.set_title("5-fold CV on training set")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def top_features(model, n=20):
    """Most influential features of a linear model (what the model 'learned')"""
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None
    names = model.named_steps["features"].get_feature_names_out()
    coefs = np.ravel(clf.coef_)
    order = np.argsort(coefs)
    return {
        "judol": [(names[i], float(coefs[i])) for i in order[::-1][:n]],
        "normal": [(names[i], float(coefs[i])) for i in order[:n]],
    }


def error_analysis(model, X_test, y_test):
    """Misclassified test comments, to read and learn from"""
    predictions = model.predict(X_test)
    df = pd.DataFrame({"komentar": X_test, "label": y_test, "predicted": predictions})
    errors = df[df["label"] != df["predicted"]].copy()
    errors["error_type"] = np.where(errors["label"] == 1, "false_negative", "false_positive")
    return errors
