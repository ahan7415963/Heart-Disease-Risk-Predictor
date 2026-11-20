"""
train.py — Train six classifiers with GridSearchCV hyperparameter tuning.

Each model is wrapped inside a full sklearn Pipeline:
    preprocessor → classifier
so that scaling and encoding happen inside each cross-validation fold,
preventing data leakage.

Usage:
    cd heart-disease-predictor
    python -m src.train
"""

import time
import warnings
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from src.preprocess import load_data, split_data, get_preprocessor
from src.utils import RANDOM_STATE, MODEL_DIR, ensure_dirs

warnings.filterwarnings("ignore")  # suppress convergence warnings during grid search


# ---------------------------------------------------------------------------
# Model definitions + hyperparameter grids
# ---------------------------------------------------------------------------
def get_model_configs():
    """
    Return a dict of {model_name: (estimator, param_grid)}.

    param_grid keys are prefixed with 'clf__' because the estimator sits
    inside a Pipeline under the name 'clf'.
    """
    configs = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "clf__max_depth": [3, 5, 7, None],
                "clf__min_samples_split": [2, 5, 10],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [5, 10, None],
            },
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", 0.01, 0.1],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 7, 9, 11]},
        ),
    }
    return configs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_all_models(X_train, y_train):
    """
    Train every model using GridSearchCV (5-fold stratified CV).

    Returns
    -------
    results : dict
        {model_name: {"pipeline": best Pipeline,
                       "best_params": dict,
                       "cv_score_mean": float,
                       "cv_score_std": float,
                       "train_time": float}}
    """
    preprocessor = get_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    configs = get_model_configs()
    for name, (estimator, param_grid) in configs.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        # Build a full pipeline: preprocess → classify
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", estimator),
        ])

        grid = GridSearchCV(
            pipe,
            param_grid,
            cv=cv,
            scoring="f1",          # optimize for F1 (medical context: recall matters)
            n_jobs=-1,
            return_train_score=True,
        )

        start = time.time()
        grid.fit(X_train, y_train)
        elapsed = time.time() - start

        results[name] = {
            "pipeline": grid.best_estimator_,
            "best_params": grid.best_params_,
            "cv_score_mean": grid.best_score_,
            "cv_score_std": grid.cv_results_["std_test_score"][grid.best_index_],
            "train_time": elapsed,
        }

        print(f"  Best params : {grid.best_params_}")
        print(f"  CV F1 (mean): {grid.best_score_:.4f} ± {results[name]['cv_score_std']:.4f}")
        print(f"  Train time  : {elapsed:.2f}s")

    return results


def select_best_model(results):
    """Pick the model with the highest cross-validated F1 score."""
    best_name = max(results, key=lambda k: results[k]["cv_score_mean"])
    return best_name, results[best_name]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    ensure_dirs()

    # 1. Load and split
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test  set: {X_test.shape[0]} samples")

    # 2. Train all models
    results = train_all_models(X_train, y_train)

    # 3. Select best model
    best_name, best_info = select_best_model(results)
    print(f"\n{'='*60}")
    print(f"★ Best model (by CV F1): {best_name}")
    print(f"  CV F1 = {best_info['cv_score_mean']:.4f} ± {best_info['cv_score_std']:.4f}")
    print(f"{'='*60}")

    # 4. Save the best pipeline
    model_path = f"{MODEL_DIR}/best_model.joblib"
    joblib.dump(best_info["pipeline"], model_path)
    print(f"\n✓ Best model saved to {model_path}")

    # 5. Also save results dict for evaluate.py
    joblib.dump(results, f"{MODEL_DIR}/all_results.joblib")
    print(f"✓ All results saved to {MODEL_DIR}/all_results.joblib")

    return results, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
