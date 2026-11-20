"""
app.py — Streamlit web application for Heart Disease Risk Prediction.

Launch with:
    cd heart-disease-predictor
    streamlit run app/app.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Setup — resolve project root so we can load the model
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
METRICS_PATH = os.path.join(PROJECT_ROOT, "reports", "metrics.csv")
ROC_PATH = os.path.join(PROJECT_ROOT, "reports", "figures", "roc_curves.png")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load model (cached so it isn't reloaded on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🩺 Predict", "📊 Model Comparison"])

# ===================== TAB 1: PREDICTION =====================
with tab1:
    st.title("❤️ Heart Disease Risk Predictor")
    st.divider()

    # --- Input form in two columns ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Information")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=55,
                              help="Patient's age in years")
        sex = st.selectbox("Sex", options=[1, 0],
                           format_func=lambda x: "Male" if x == 1 else "Female",
                           help="1 = Male, 0 = Female")
        cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3],
                          format_func=lambda x: {
                              0: "0 — Typical Angina",
                              1: "1 — Atypical Angina",
                              2: "2 — Non-Anginal Pain",
                              3: "3 — Asymptomatic"
                          }[x],
                          help="Type of chest pain experienced")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)",
                                   min_value=80, max_value=220, value=130,
                                   help="Resting blood pressure on admission")
        chol = st.number_input("Serum Cholesterol (mg/dl)",
                               min_value=100, max_value=600, value=240,
                               help="Serum cholesterol in mg/dl")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)",
                           options=[0, 1],
                           format_func=lambda x: "Yes" if x == 1 else "No",
                           help="1 if fasting blood sugar > 120 mg/dl, else 0")

    with col2:
        st.subheader("Clinical Measurements")
        restecg = st.selectbox("Resting ECG Result (restecg)", options=[0, 1, 2],
                               format_func=lambda x: {
                                   0: "0 — Normal",
                                   1: "1 — ST-T Wave Abnormality",
                                   2: "2 — Left Ventricular Hypertrophy"
                               }[x],
                               help="Resting electrocardiographic results")
        thalach = st.number_input("Max Heart Rate Achieved (thalach)",
                                  min_value=60, max_value=220, value=150,
                                  help="Maximum heart rate during exercise test")
        exang = st.selectbox("Exercise-Induced Angina (exang)",
                             options=[0, 1],
                             format_func=lambda x: "Yes" if x == 1 else "No",
                             help="1 = yes, 0 = no")
        oldpeak = st.number_input("ST Depression (oldpeak)",
                                  min_value=0.0, max_value=7.0, value=1.0,
                                  step=0.1,
                                  help="ST depression induced by exercise relative to rest")
        slope = st.selectbox("Slope of Peak Exercise ST Segment",
                             options=[0, 1, 2],
                             format_func=lambda x: {
                                 0: "0 — Upsloping",
                                 1: "1 — Flat",
                                 2: "2 — Downsloping"
                             }[x],
                             help="Slope of the peak exercise ST segment")
        ca = st.selectbox("Number of Major Vessels Colored (ca)",
                          options=[0, 1, 2, 3],
                          help="Number of major vessels (0-3) colored by fluoroscopy")
        thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3],
                            format_func=lambda x: {
                                0: "0 — Normal",
                                1: "1 — Fixed Defect",
                                2: "2 — Reversible Defect",
                                3: "3 — Other"
                            }[x],
                            help="Blood disorder status from thallium stress test")

    st.divider()

    # --- Predict button ---
    if st.button("🔍 Predict", use_container_width=True, type="primary"):
        model = load_model()

        # Build input DataFrame with the same column order the pipeline expects
        input_data = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope,
            "ca": ca, "thal": thal,
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.divider()

        # --- Results ---
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("### ⚠️ Higher Risk of Heart Disease")
                st.metric("Prediction", "Positive (Heart Disease)")
            else:
                st.success("### ✅ Lower Risk of Heart Disease")
                st.metric("Prediction", "Negative (No Heart Disease)")

        with result_col2:
            prob_disease = probability[1] * 100
            st.metric("Heart Disease Probability", f"{prob_disease:.1f}%")

            # Progress bar as visual gauge
            st.progress(probability[1], text=f"Risk: {prob_disease:.1f}%")

        # Show the input data for transparency
        with st.expander("📋 View input data sent to model"):
            st.dataframe(input_data)


# ===================== TAB 2: MODEL COMPARISON =====================
with tab2:
    st.title("📊 Model Comparison")

    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)

        # Clean display — format numbers
        display_cols = ["Model", "Accuracy", "Precision", "Recall",
                        "F1-Score", "ROC-AUC", "CV Accuracy (mean)",
                        "CV Accuracy (std)", "Train Time (s)"]
        available_cols = [c for c in display_cols if c in metrics_df.columns]
        st.dataframe(
            metrics_df[available_cols].style.format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}",
                "Recall": "{:.4f}", "F1-Score": "{:.4f}",
                "ROC-AUC": "{:.4f}", "CV Accuracy (mean)": "{:.4f}",
                "CV Accuracy (std)": "{:.4f}",
            }),
            use_container_width=True,
        )

        # Highlight best model
        best_row = metrics_df.iloc[0]
        st.success(
            f"**Best Model:** {best_row['Model']} — "
            f"F1 = {best_row['F1-Score']:.4f}, "
            f"ROC-AUC = {best_row['ROC-AUC']:.4f}"
        )
    else:
        st.warning("metrics.csv not found. Run training first.")

    st.divider()
    st.subheader("ROC Curves")
    if os.path.exists(ROC_PATH):
        st.image(ROC_PATH, caption="ROC Curves — All Models",
                 use_container_width=True)
    else:
        st.warning("ROC curves not generated yet. Run evaluation first.")
