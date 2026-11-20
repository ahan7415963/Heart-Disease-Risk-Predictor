"""
evaluate.py — Evaluate all trained models on the held-out test set.

Produces:
  • reports/metrics.csv           — accuracy, precision, recall, F1, ROC-AUC
  • reports/figures/confusion_matrices.png
  • reports/figures/best_cm.png
  • reports/figures/roc_curves.png
  • reports/figures/feature_importance.png

Usage:
    cd heart-disease-predictor
    python -m src.evaluate
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.preprocess import load_data, split_data
from src.utils import RANDOM_STATE, MODEL_DIR, REPORTS_DIR, FIGURES_DIR, ensure_dirs


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(results, X_train, X_test, y_train, y_test):
    """
    Compute test-set metrics + 5-fold CV accuracy for every model.

    Returns a DataFrame with one row per model.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, info in results.items():
        pipe = info["pipeline"]
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        # Cross-validated accuracy on the TRAINING set (shows stability)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                    scoring="accuracy", n_jobs=-1)

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "CV Accuracy (mean)": cv_scores.mean(),
            "CV Accuracy (std)": cv_scores.std(),
            "Best Params": str(info["best_params"]),
            "Train Time (s)": round(info["train_time"], 2),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("F1-Score", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_confusion_matrices(results, X_test, y_test, save_path=None):
    """Plot a 2×3 grid of confusion matrices for all six models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (name, info) in enumerate(results.items()):
        y_pred = info["pipeline"].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"],
                    ax=axes[i])
        axes[i].set_title(name, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_best_cm(best_pipeline, X_test, y_test, best_name, save_path=None):
    """Plot a single large confusion matrix for the best model."""
    y_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"],
                annot_kws={"size": 16}, ax=ax)
    ax.set_title(f"Best Model: {best_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(results, X_test, y_test, save_path=None):
    """Plot ROC curves for all models on one chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, info in results.items():
        y_proba = info["pipeline"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(results, X_train, save_path=None):
    """
    Plot feature importances from the Random Forest model.

    The RF stores importances for the *transformed* features (after one-hot
    encoding), so we extract feature names from the preprocessor.
    """
    rf_pipeline = results["Random Forest"]["pipeline"]
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    rf_model = rf_pipeline.named_steps["clf"]

    # Get transformed feature names
    num_features = preprocessor.transformers_[0][2]  # numeric feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_features = list(cat_encoder.get_feature_names_out())
    all_names = list(num_features) + cat_features

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # top 15

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices][::-1], color="steelblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([all_names[i] for i in indices][::-1])
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Random Forest — Top Feature Importances", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    ensure_dirs()

    # Load data and results
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    results = joblib.load(os.path.join(MODEL_DIR, "all_results.joblib"))

    # 1. Compute metrics table
    metrics_df = compute_metrics(results, X_train, X_test, y_train, y_test)
    csv_path = os.path.join(REPORTS_DIR, "metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print("✓ Metrics saved to reports/metrics.csv")
    print(metrics_df.to_string(index=False))

    # 2. Find the best model (by CV F1 from training)
    best_name = max(results, key=lambda k: results[k]["cv_score_mean"])
    best_pipeline = results[best_name]["pipeline"]

    # 3. Generate all plots
    plot_confusion_matrices(
        results, X_test, y_test,
        save_path=os.path.join(FIGURES_DIR, "confusion_matrices.png"),
    )
    plot_best_cm(
        best_pipeline, X_test, y_test, best_name,
        save_path=os.path.join(FIGURES_DIR, "best_cm.png"),
    )
    plot_roc_curves(
        results, X_test, y_test,
        save_path=os.path.join(FIGURES_DIR, "roc_curves.png"),
    )
    plot_feature_importance(
        results, X_train,
        save_path=os.path.join(FIGURES_DIR, "feature_importance.png"),
    )
    print("✓ All figures saved to reports/figures/")

    return metrics_df, best_name


if __name__ == "__main__":
    main()
