"""
tests/test_project.py — Automated tests for the Heart Disease Predictor.

Run with:
    cd heart-disease-predictor
    python -m pytest tests/ -v
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import DATA_PATH, MODEL_DIR
from src.preprocess import load_data, split_data, get_preprocessor


# ---------------------------------------------------------------------------
# Test 1: Dataset loads with expected shape
# ---------------------------------------------------------------------------
def test_dataset_shape():
    """The raw dataset should have 303 rows and 14 columns."""
    df = pd.read_csv(DATA_PATH)
    assert df.shape == (303, 14), f"Expected (303, 14), got {df.shape}"


# ---------------------------------------------------------------------------
# Test 2: No missing values after preprocessing
# ---------------------------------------------------------------------------
def test_no_missing_after_preprocessing():
    """The pipeline should produce zero NaN values."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = get_preprocessor()
    X_transformed = preprocessor.fit_transform(X_train)
    # ColumnTransformer returns a numpy array (or sparse matrix)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    assert np.isnan(X_transformed).sum() == 0, "Found NaN values after preprocessing"


# ---------------------------------------------------------------------------
# Test 3: Saved model loads and predicts 0 or 1
# ---------------------------------------------------------------------------
def test_model_prediction():
    """The best model should load and return 0 or 1 for a sample input."""
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet — run src.train first")

    model = joblib.load(model_path)

    # Sample input (a typical patient)
    sample = pd.DataFrame([{
        "age": 55, "sex": 1, "cp": 2, "trestbps": 130,
        "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
        "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2,
    }])

    pred = model.predict(sample)[0]
    assert pred in [0, 1], f"Prediction should be 0 or 1, got {pred}"


# ---------------------------------------------------------------------------
# Test 4: Predicted probabilities are between 0 and 1
# ---------------------------------------------------------------------------
def test_prediction_probabilities():
    """predict_proba should return values in [0, 1]."""
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet — run src.train first")

    model = joblib.load(model_path)

    sample = pd.DataFrame([{
        "age": 45, "sex": 0, "cp": 1, "trestbps": 120,
        "chol": 200, "fbs": 0, "restecg": 0, "thalach": 170,
        "exang": 0, "oldpeak": 0.5, "slope": 2, "ca": 0, "thal": 2,
    }])

    proba = model.predict_proba(sample)
    assert proba.shape == (1, 2), f"Expected shape (1, 2), got {proba.shape}"
    assert np.all(proba >= 0) and np.all(proba <= 1), \
        f"Probabilities should be in [0, 1], got {proba}"
    assert abs(proba.sum() - 1.0) < 1e-6, \
        f"Probabilities should sum to 1, got {proba.sum()}"
