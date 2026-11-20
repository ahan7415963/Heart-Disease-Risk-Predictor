# Presentation Slide Outline — Heart Disease Risk Predictor

## Slide 1: Title Slide
- **Title:** Heart Disease Risk Predictor — ML Classification Comparison
- **Subtitle:** Machine Learning Mini Project
- **Name, Roll Number, Course:** B.Tech CSE (AI & Data Science), Semester V

---

## Slide 2: Problem Statement
- Heart disease is a leading cause of death worldwide
- Early detection saves lives — can we predict heart disease from routine clinical measurements?
- **Goal:** Build and compare ML classifiers, deploy the best one as a web app
- Dataset: UCI Heart Disease (Cleveland) — 303 patients, 13 features, binary target

---

## Slide 3: Dataset & EDA Highlights
- 303 rows, 13 features (5 numeric, 8 categorical), 1 binary target
- No missing values; 1 duplicate removed
- Roughly balanced: 54% disease vs 46% no disease
- Key predictors: chest pain type, max heart rate, exercise-induced angina, ST depression
- *Show:* target distribution bar chart + correlation heatmap

---

## Slide 4: Methodology
- **Preprocessing:** StandardScaler (numeric) + OneHotEncoder (categorical) inside a Pipeline to prevent data leakage
- **Split:** 80/20 stratified train/test (random_state=42)
- **Models:** Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM (RBF), KNN
- **Tuning:** GridSearchCV with 5-fold StratifiedKFold, optimized on F1-score
- *Show:* pipeline diagram (Preprocess → Classify → Evaluate)

---

## Slide 5: Results — Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| **Logistic Regression** | **0.869** | **0.838** | **0.939** | **0.886** | **0.905** |
| SVM | 0.820 | 0.789 | 0.909 | 0.845 | 0.891 |
| Random Forest | 0.787 | 0.750 | 0.909 | 0.822 | 0.918 |
| KNN | 0.770 | 0.757 | 0.848 | 0.800 | 0.869 |
| Decision Tree | 0.738 | 0.718 | 0.848 | 0.778 | 0.774 |
| Naive Bayes | 0.738 | 0.840 | 0.636 | 0.724 | 0.874 |

- *Show:* ROC curve comparison chart

---

## Slide 6: Best Model & Why
- **Logistic Regression** (C = 0.1) selected as the best model
- **Why?**
  1. Highest test F1-score (0.886) — best precision-recall balance
  2. Recall = 93.9% — catches nearly all disease cases (critical in medical screening)
  3. Interpretable — coefficients show feature importance directly
  4. Generalizes well — stable cross-validation performance
- In medical screening, **missing a true positive (false negative) is worse than a false alarm**
- *Show:* confusion matrix for Logistic Regression

---

## Slide 7: Live Demo
- **Streamlit App** — enter 13 patient features, get instant prediction
- Shows: predicted class, probability, risk gauge
- Also includes model comparison tab with metrics table and ROC curves
- *Run:* `streamlit run app/app.py`

---

## Slide 8: Limitations & Future Work
- **Limitations:**
  - Small dataset (303 samples) — results may not generalize
  - Old dataset (1988) — missing modern biomarkers (troponin, BNP)
  - Single hospital (Cleveland Clinic) — potential population bias
- **Future Work:**
  - Use larger datasets (MIMIC-III, UK Biobank)
  - Try gradient boosting (XGBoost, LightGBM)
  - Add SHAP explainability
  - Multi-class severity prediction
- **Thank you! Questions?**
