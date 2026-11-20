# Demo Script — Heart Disease Risk Predictor (2 Minutes)

## Opening (15 seconds)
"Good morning/afternoon. I'm presenting my ML mini-project: a Heart Disease Risk Predictor. The goal is to predict whether a patient is at risk of heart disease using clinical measurements, and deploy the best model as a web app."

## Dataset & EDA (25 seconds)
"I used the UCI Heart Disease dataset — 303 patients with 13 clinical features like age, blood pressure, cholesterol, and chest pain type. The target is binary: disease or no disease. The dataset is roughly balanced at 54-46%, has no missing values, and I found that chest pain type, max heart rate, and exercise-induced angina are the strongest predictors."

## Methodology (25 seconds)
"I compared six models: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM, and KNN. All preprocessing — standard scaling for numeric features and one-hot encoding for categorical features — is inside a scikit-learn Pipeline to prevent data leakage. I used GridSearchCV with 5-fold stratified cross-validation to tune hyperparameters, optimizing for F1-score because in medical screening, we need both precision and recall."

## Results (25 seconds)
"As you can see in the results table, Logistic Regression performed best with an F1-score of 0.886 and recall of 93.9% — meaning it correctly identifies 31 out of 33 disease cases. I chose it over KNN, which had slightly higher cross-validation F1 but performed worse on unseen data, suggesting overfitting. Logistic Regression is also the most interpretable model — its coefficients directly tell us which features increase disease risk."

## Live Demo (20 seconds)
"Let me show you the Streamlit app. Here I can enter patient measurements — say a 55-year-old male with chest pain, high blood pressure — click Predict, and it shows the risk level with a probability score. The Model Comparison tab displays the full metrics table and ROC curves."

## Closing (10 seconds)
"The main limitation is the small, dated dataset. Future work could include larger datasets and explainability tools like SHAP. Thank you — any questions?"
