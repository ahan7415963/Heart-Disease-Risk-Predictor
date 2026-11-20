"""
demo.py — Terminal Demonstration Script for VIVA & Teacher Evaluation.

Demonstrates:
  1. Pipeline verification and hyperparameter inspection
  2. Live predictions on sample clinical patient profiles
  3. Feature weighting & dataset nuances
  4. Final model performance summary from reports/metrics.csv

Run with:
    cd heart-disease-predictor
    python demo.py
"""

import os
import sys
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
METRICS_PATH = os.path.join(PROJECT_ROOT, "reports", "metrics.csv")


def print_banner(text):
    print("\n" + "=" * 72)
    print(f"  {text}")
    print("=" * 72)


def run_terminal_demo():
    print_banner("HEART DISEASE RISK PREDICTOR — TERMINAL DEMONSTRATION")
    
    # 1. Model Inspection
    print(f"[1] Loading Saved Pipeline from: models/best_model.joblib")
    if not os.path.exists(MODEL_PATH):
        print("[-] Model file not found! Please run 'python -m src.train' first.")
        return

    model = joblib.load(MODEL_PATH)
    print("    [✓] Model successfully loaded.")
    print("    [i] Pipeline Architecture:")
    for step_name, step_obj in model.named_steps.items():
        print(f"        • {step_name}: {step_obj.__class__.__name__}")
    
    clf = model.named_steps["clf"]
    print(f"    [i] Classifier: {clf.__class__.__name__} with parameters:")
    print(f"        C = {clf.C}, max_iter = {clf.max_iter}, random_state = {clf.random_state}")

    # 2. Live Patient Predictions
    print_banner("[2] LIVE PATIENT INFERENCE (Testing End-to-End Pipeline)")

    test_cases = [
        {
            "Profile": "Sample Case 1 (Non-anginal CP, High Max HR, Low ST Depression)",
            "Data": {
                "age": 52,
                "sex": 1,
                "cp": 2,            # Non-anginal pain
                "trestbps": 124,
                "chol": 230,
                "fbs": 0,
                "restecg": 1,
                "thalach": 168,     # High exercise heart rate
                "exang": 0,
                "oldpeak": 0.4,
                "slope": 2,
                "ca": 0,
                "thal": 2
            }
        },
        {
            "Profile": "Sample Case 2 (Typical CP, Low Max HR, Severe ST Depression, 2 Major Vessels)",
            "Data": {
                "age": 60,
                "sex": 1,
                "cp": 0,            # Typical angina
                "trestbps": 145,
                "chol": 282,
                "fbs": 1,
                "restecg": 0,
                "thalach": 120,     # Blunted heart rate
                "exang": 1,         # Exercise-induced angina
                "oldpeak": 2.8,     # Pronounced ST depression
                "slope": 1,
                "ca": 2,            # 2 calcified vessels
                "thal": 3
            }
        }
    ]

    for tc in test_cases:
        name = tc["Profile"]
        df_patient = pd.DataFrame([tc["Data"]])
        
        # Inference through complete pipeline
        pred = model.predict(df_patient)[0]
        proba = model.predict_proba(df_patient)[0]

        print(f"\n>>> {name}")
        print("    Input Features:")
        print(f"      Age: {tc['Data']['age']} | Sex: {'Male' if tc['Data']['sex']==1 else 'Female'} | CP: Type {tc['Data']['cp']} | BP: {tc['Data']['trestbps']} mmHg")
        print(f"      Cholesterol: {tc['Data']['chol']} mg/dL | Max HR (thalach): {tc['Data']['thalach']} | ST Dep (oldpeak): {tc['Data']['oldpeak']}")
        print(f"      Exercise Angina (exang): {tc['Data']['exang']} | Colored Vessels (ca): {tc['Data']['ca']}")
        
        print("    Pipeline Prediction:")
        print(f"      Predicted Class : {pred} ({'Target = 1 (Disease Risk)' if pred == 1 else 'Target = 0 (No Disease / Healthy)'})")
        print(f"      Class 0 Prob    : {proba[0]*100:.2f}%")
        print(f"      Class 1 Prob    : {proba[1]*100:.2f}%")
        if pred == 1:
            print("      Clinical Verdict: ⚠️ Higher Risk of Heart Disease")
        else:
            print("      Clinical Verdict: ✅ Lower Risk of Heart Disease")

    # 3. Model Benchmark Summary
    print_banner("[3] SIX-MODEL BENCHMARK RESULTS (from reports/metrics.csv)")
    if os.path.exists(METRICS_PATH):
        df_metrics = pd.read_csv(METRICS_PATH)
        display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        print(df_metrics[display_cols].to_string(index=False))
    else:
        print("[-] metrics.csv not found.")

    print_banner("DEMO COMPLETED SUCCESSFULLY — READY FOR PRESENTATION")


if __name__ == "__main__":
    run_terminal_demo()
