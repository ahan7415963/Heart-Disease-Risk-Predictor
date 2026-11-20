"""
utils.py — Shared constants and helper functions used across the project.

This module centralizes configuration (paths, random seed, feature lists)
so every other module imports from one place.  Changing a path or adding
a feature only needs to happen here.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility — every random operation uses this seed
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Paths (relative to the project root: heart-disease-predictor/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "heart.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
# Numeric features — will be scaled with StandardScaler
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Categorical features — will be one-hot encoded
# These columns are stored as integers in the CSV (e.g., cp: 0–3) but they
# represent *categories*, not ordered quantities.  For example, cp (chest pain
# type) has values {0: typical angina, 1: atypical angina, 2: non-anginal,
# 3: asymptomatic}.  Treating them as numeric would wrongly imply an ordering
# and magnitude relationship, so we one-hot encode them instead.
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Binary target column
TARGET = "target"

# All feature columns (order matters for ColumnTransformer)
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [MODEL_DIR, REPORTS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
