
import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import pickle
from src.data_ingestion import load_and_process_data
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def run_productive_audit():
    print("=== Integration Test: Productive Flow (Mock) ===")

    # 1. Load Data
    raw_path = "data/01_raw/LLCP2022_10rows.xpt"
    if not os.path.exists(raw_path):
        print("Data not found!")
        return

    df = load_and_process_data(raw_path, "data/02_intermediate")
    if df is None or df.empty:
        print("Data loading failed.")
        return

    target = 'CVDINFR4'
    X = df.drop(columns=[target])
    y = df[target]

    # Mocking PyCaret's work by using standard sklearn
    # In the real flow, PyCaret would handle imputation/encoding/etc.
    # Here we just want to prove we can train -> save -> load -> predict

    # Simple preprocessing: drop non-numeric for this mock
    X_num = X.select_dtypes(include=[np.number]).fillna(0)

    print("\nTraining Mock Model (RandomForest)...")
    pipeline = Pipeline([
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    pipeline.fit(X_num, y)
    print("Training complete.")

    # 2. Save Model (Serialization)
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "productive_model.pkl")

    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_path}")

    # 3. Load Model & Predict
    print("\nLoading Model...")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    print("Predicting on new data...")
    # Take first 3 rows as "new users"
    sample = X_num.iloc[:3]
    preds = loaded_model.predict(sample)
    probs = loaded_model.predict_proba(sample)[:, 1]

    print(f"Sample Input (SEQNO): {sample.index.tolist()}")
    print(f"Predictions: {preds}")
    print(f"Probabilities: {probs}")

    print("\n=== Productive Flow Audit Successful ===")

if __name__ == "__main__":
    run_productive_audit()
