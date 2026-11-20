"""
eda.py — Exploratory Data Analysis (script version of the notebook).

Generates all EDA figures and prints summary statistics.
Run from the project root:
    cd heart-disease-predictor
    python -m src.eda
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import DATA_PATH, FIGURES_DIR, ensure_dirs


def run_eda():
    ensure_dirs()
    df = pd.read_csv(DATA_PATH)

    # -----------------------------------------------------------------------
    # Basic info
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicates: {df.duplicated().sum()}")

    # Remove duplicates if any
    if df.duplicated().sum() > 0:
        print(f"  → Dropping {df.duplicated().sum()} duplicate rows.")
        df = df.drop_duplicates().reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Class balance
    # -----------------------------------------------------------------------
    print(f"\nTarget distribution:")
    print(df["target"].value_counts())
    print(f"  Class ratio (disease/total): {df['target'].mean():.2%}")

    # -----------------------------------------------------------------------
    # Figure 1: Target distribution
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["target"].value_counts()
    bars = ax.bar(["No Disease (0)", "Disease (1)"], counts.values,
                  color=["#3498db", "#e74c3c"])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Target Variable Distribution", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/target_distribution.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("✓ Saved target_distribution.png")

    # -----------------------------------------------------------------------
    # Figure 2: Correlation heatmap
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("✓ Saved correlation_heatmap.png")

    # -----------------------------------------------------------------------
    # Figure 3: Distributions of key numeric features split by target
    # -----------------------------------------------------------------------
    key_features = ["age", "chol", "thalach", "trestbps", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, feat in enumerate(key_features):
        sns.histplot(data=df, x=feat, hue="target", kde=True,
                     palette={0: "#3498db", 1: "#e74c3c"},
                     ax=axes[i], stat="density", common_norm=False)
        axes[i].set_title(f"{feat} by Target", fontsize=12, fontweight="bold")
        axes[i].legend(["No Disease", "Disease"])

    axes[-1].axis("off")  # hide the 6th subplot
    plt.suptitle("Feature Distributions by Heart Disease Status",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/feature_distributions.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("✓ Saved feature_distributions.png")

    # -----------------------------------------------------------------------
    # Figure 4: Box plots for outlier detection
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    for i, feat in enumerate(key_features):
        sns.boxplot(data=df, y=feat, x="target", palette="Set2", ax=axes[i])
        axes[i].set_title(feat, fontsize=12, fontweight="bold")
        axes[i].set_xticklabels(["No Disease", "Disease"])

    plt.suptitle("Box Plots — Outlier Detection", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved boxplots.png")

    return df


if __name__ == "__main__":
    run_eda()
