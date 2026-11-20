# ❤️ Heart Disease Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5.2-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.40.2-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/tests-4%20passed-brightgreen?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, presentation-ready Machine Learning system that predicts cardiovascular disease risk using the **UCI Heart Disease (Cleveland)** dataset. The project benchmarks six supervised classification models using strict leakage-free preprocessing and cross-validation, selects the optimal classifier based on medical screening criteria (maximizing Recall & F1), and serves real-time predictions through an interactive **Streamlit** web application and CLI terminal interface.

> **Academic Course:** Machine Learning (B.Tech CSE — AI & Data Science, Semester V)  
> **Author:** [ahan7415963](https://github.com/ahan7415963)  
> **Repository:** [Heart-Disease-Risk-Predictor](https://github.com/ahan7415963/Heart-Disease-Risk-Predictor)

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Architecture & Leakage Prevention](#-architecture--leakage-prevention)
4. [Model Comparison & Benchmark](#-model-comparison--benchmark)
5. [Key Clinical Findings](#-key-clinical-findings)
6. [Interactive Streamlit Web App](#-interactive-streamlit-web-app)
7. [Terminal Demonstration](#-terminal-demonstration)
8. [Automated Testing](#-automated-testing)
9. [Project Structure](#-project-structure)
10. [Quickstart & Installation](#-quickstart--installation)

---

## 📋 Project Overview

- **Dataset:** UCI Cleveland Heart Disease (303 patient records, 13 clinical features, binary target).
- **Target Variable:** Binary heart disease diagnosis (`0 = No Disease`, `1 = Heart Disease Risk`).
- **Models Evaluated:**
  1. Logistic Regression (Tuned $L_2$ regularization)
  2. Support Vector Machine (RBF kernel, calibrated probabilities)
  3. Random Forest (Ensemble trees with depth control)
  4. K-Nearest Neighbors (Distance-based neighbor vote)
  5. Decision Tree (CART algorithm with depth tuning)
  6. Gaussian Naive Bayes (Probabilistic baseline)
- **Best Model:** **Logistic Regression** ($C = 0.1$) — Test F1: **0.8857**, Recall: **93.94%**, ROC-AUC: **0.9048**.

---

## 📊 Dataset Description

The dataset consists of 303 clinical records with 13 predictive attributes:

| Feature | Type | Description | Values / Units |
| :--- | :--- | :--- | :--- |
| `age` | Numeric | Patient age | Years (29 – 77) |
| `sex` | Categorical | Biological sex | 1 = Male, 0 = Female |
| `cp` | Categorical | Chest pain type | 0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic |
| `trestbps` | Numeric | Resting blood pressure | mm Hg on admission (94 – 200) |
| `chol` | Numeric | Serum cholesterol | mg/dl (126 – 564) |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl | 1 = True, 0 = False |
| `restecg` | Categorical | Resting electrocardiographic results | 0: Normal, 1: ST-T wave abnormality, 2: LV hypertrophy |
| `thalach` | Numeric | Maximum heart rate achieved | bpm during thallium stress test (71 – 202) |
| `exang` | Categorical | Exercise-induced angina | 1 = Yes, 0 = No |
| `oldpeak` | Numeric | ST depression induced by exercise | Relative to rest (0.0 – 6.2) |
| `slope` | Categorical | Slope of peak exercise ST segment | 0: Upsloping, 1: Flat, 2: Downsloping |
| `ca` | Categorical | Major vessels colored by fluoroscopy | 0 – 3 vessels |
| `thal` | Categorical | Thallium stress test defect | 1: Normal, 2: Fixed defect, 3: Reversible defect |

---

## 🛠️ Architecture & Leakage Prevention

To ensure zero data leakage between training, cross-validation, and test partitions:

```
[ Raw Patient Data (heart.csv) ]
               │
               ▼
   [ Stratified 80/20 Train/Test Split (random_state=42) ]
               │
   ┌───────────┴────────────────────────────────────────┐
   │ (Training Set: 242 samples)                        │ (Test Set: 61 samples)
   ▼                                                    │ [Held Out Until Final Eval]
[ ColumnTransformer Pipeline ]                          │
 ├── Numeric Features     ─► StandardScaler()           │
 └── Categorical Features ─► OneHotEncoder()            │
   │                                                    │
   ▼                                                    │
[ 5-Fold StratifiedKFold GridSearchCV ]                 │
 └── Optimizes hyperparameters on CV F1-score           │
   │                                                    │
   ▼                                                    │
[ Best Pipeline Saved: models/best_model.joblib ]       │
   │                                                    │
   └──────────────────────┬─────────────────────────────┘
                          ▼
            [ Final Unbiased Test Evaluation ]
             ├── Accuracy, Precision, Recall, F1, ROC-AUC
             └── Confusion Matrices & ROC Curves
```

- **Numeric columns** are scaled via `StandardScaler` only on the training folds.
- **Categorical columns** (`cp`, `restecg`, `slope`, `ca`, `thal`, etc.) are one-hot encoded using `OneHotEncoder(handle_unknown='ignore')` so no artificial ordinal distance is imposed.

---

## 🏆 Model Comparison & Benchmark

Evaluated on the held-out stratified test set (61 patient samples):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV Accuracy (Train) | Best Hyperparameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **0.8689** | **0.8378** | **0.9394** | **0.8857** | **0.9048** | $0.8264 \pm 0.0666$ | `{'C': 0.1}` |
| **SVM (RBF Kernel)** | 0.8197 | 0.7895 | 0.9091 | 0.8451 | 0.8907 | $0.8263 \pm 0.0731$ | `{'C': 0.1, 'gamma': 0.1}` |
| **Random Forest** | 0.7869 | 0.7500 | 0.9091 | 0.8219 | 0.9177 | $0.8138 \pm 0.0804$ | `{'max_depth': 5, 'n_estimators': 100}` |
| **KNN** | 0.7705 | 0.7568 | 0.8485 | 0.8000 | 0.8690 | $0.8388 \pm 0.0361$ | `{'n_neighbors': 5}` |
| **Decision Tree** | 0.7377 | 0.7179 | 0.8485 | 0.7778 | 0.7744 | $0.7685 \pm 0.0715$ | `{'max_depth': 3, 'min_samples_split': 2}` |
| **Gaussian Naive Bayes** | 0.7377 | 0.8400 | 0.6364 | 0.7241 | 0.8745 | $0.7683 \pm 0.0757$ | `{'var_smoothing': 1e-07}` |

### Why Recall Matters in Medical Diagnosis
In cardiac screening, a **False Negative** (sending a diseased patient home without treatment) is clinically unacceptable. Logistic Regression achieves **93.94% recall**, missing only 2 cases out of 33 diseased patients while maintaining 83.78% precision and full mathematical interpretability.

---

## 📈 Key Clinical Findings

1. **Chest Pain (`cp`) & Maximum Heart Rate (`thalach`):** Patients presenting with non-anginal or atypical pain and higher stress test heart rates had higher probabilities of positive diagnostic classification.
2. **ST Depression (`oldpeak`):** Marked ST-segment depression induced by exercise strongly indicates coronary ischemia and absence of healthy arterial reserve.
3. **Number of Major Vessels (`ca`):** Fluoroscopy-colored vessels correlate inversely with healthy myocardial perfusion.

All visualization charts are generated and saved under `reports/figures/`:
- `target_distribution.png` — Class balance check
- `correlation_heatmap.png` — Feature cross-correlations
- `feature_distributions.png` — Distribution of continuous clinical variables
- `boxplots.png` — Outlier identification
- `confusion_matrices.png` — 2×3 grid of confusion matrices across all six models
- `best_cm.png` — Best model confusion matrix
- `roc_curves.png` — Combined ROC curves with AUC scores
- `feature_importance.png` — Random Forest feature importance hierarchy

---

## 🖥️ Interactive Streamlit Web App

The project includes an interactive web interface built with Streamlit:

```bash
streamlit run app/app.py
```

### Features:
- **Clinical Inputs:** Sliders and selection boxes with physiological ranges and tooltip descriptions for all 13 patient attributes.
- **Risk Assessment:** Instant classification output, numerical disease probability percentage, and an intuitive risk gauge.
- **Model Comparison Tab:** Embedded benchmark comparison table and high-resolution ROC curves.

---

## 💻 Terminal Demonstration

For direct command-line viva evaluation:

```bash
python demo.py
```

Executes model loading, inspects pipeline architecture, and evaluates end-to-end inference on contrasting clinical patient profiles (low-risk vs. high-risk) with detailed probability breakdowns.

---

## 🧪 Automated Testing

Verify code correctness and pipeline stability via pytest:

```bash
python -m pytest tests/ -v
```

Automated test cases include:
- `test_dataset_shape`: Validates expected dimensions (303 rows, 14 columns).
- `test_no_missing_after_preprocessing`: Asserts zero NaN generation post-transformation.
- `test_model_prediction`: Validates that loaded model outputs valid binary classifications (`0` or `1`).
- `test_prediction_probabilities`: Asserts that `predict_proba` values fall strictly in $[0, 1]$ and sum to $1.0$.

---

## 📁 Project Structure

```
heart-disease-predictor/
├── .gitignore                   # Excludes caches, venvs, and temp files
├── README.md                    # Comprehensive documentation
├── requirements.txt             # Pinned dependency versions
├── demo.py                      # Terminal demonstration script for viva
├── data/
│   └── heart.csv                # Raw UCI Cleveland dataset (303 rows)
├── notebooks/
│   └── eda.ipynb                # Interactive Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Paths, constants, feature definitions
│   ├── preprocess.py            # ColumnTransformer & train/test split
│   ├── train.py                 # 6-model training with Stratified GridSearchCV
│   ├── evaluate.py              # Test set evaluation, metrics, and plots
│   └── eda.py                   # Script version of EDA
├── models/
│   ├── best_model.joblib        # Serialized best pipeline (Logistic Regression)
│   └── all_results.joblib       # Serialized training metadata & pipelines
├── reports/
│   ├── metrics.csv              # Exact quantitative model comparison table
│   ├── report.md                # 3-page academic mini-project report
│   ├── slides_outline.md        # 8-slide presentation structure
│   ├── demo_script.md           # 2-minute oral presentation walkthrough
│   ├── viva_questions.md        # 5 high-frequency teacher viva questions & answers
│   └── figures/                 # All 8 high-resolution figures (.png)
├── app/
│   └── app.py                   # Streamlit web application
└── tests/
    ├── __init__.py
    └── test_project.py          # Pytest verification suite
```

---

## 🚀 Quickstart & Installation

### 1. Clone the repository
```bash
git clone https://github.com/ahan7415963/Heart-Disease-Risk-Predictor.git
cd Heart-Disease-Risk-Predictor
```

### 2. Set up virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run scripts
```bash
# Run EDA
python -m src.eda

# Train models
python -m src.train

# Evaluate models
python -m src.evaluate

# Run test suite
python -m pytest tests/ -v

# Launch Web Application
streamlit run app/app.py
```

---

## 📜 License & Acknowledgments
- **Dataset:** UCI Machine Learning Repository / Cleveland Clinic Foundation (Dr. Robert Detrano).
- **License:** MIT License. Free for academic and educational usage.
