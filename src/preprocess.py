"""
preprocess.py — Data loading, cleaning, and sklearn preprocessing pipeline.

Key design decision: all transformations (scaling, encoding) live inside a
scikit-learn Pipeline/ColumnTransformer so they are fitted ONLY on training
data.  This prevents data leakage (the test set never influences scaling
parameters or category mappings).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.utils import (
    DATA_PATH, RANDOM_STATE,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
)


def load_data(path=None):
    """
    Load the UCI Heart Disease CSV and return (X, y).

    Parameters
    ----------
    path : str, optional
        Override the default data path (useful for tests).

    Returns
    -------
    X : pd.DataFrame  — 13-column feature matrix
    y : pd.Series      — binary target (0 = no disease, 1 = disease)
    """
    if path is None:
        path = DATA_PATH
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def get_preprocessor():
    """
    Build a ColumnTransformer that:
      • StandardScaler for numeric features  (mean=0, std=1)
      • OneHotEncoder  for categorical features (handle_unknown='ignore')

    Returns
    -------
    preprocessor : ColumnTransformer
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            # We one-hot encode even though the CSV stores integers because
            # these columns represent unordered categories (see utils.py).
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # safety — drop any extra columns
    )
    return preprocessor


def split_data(X, y, test_size=0.20):
    """
    Stratified train/test split.

    Stratification ensures the class ratio (~54 % disease / ~46 % no disease)
    is preserved in both splits, which is important for a small dataset.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ---------------------------------------------------------------------------
# Convenience: run this file directly to verify the pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X, y = load_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Missing values:\n{X.isnull().sum().sum()}")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    preprocessor = get_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    print(f"Transformed train shape: {X_train_transformed.shape}")
    print("✓ Preprocessing pipeline works correctly.")
