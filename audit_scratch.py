
import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from src.model import XGBoostScratch
from src.data_ingestion import load_and_process_data

def run_integration_test():
    print("=== Integration Test: Scratch Model ===")

    # 1. Load Data (Toy Set)
    # Re-using the ingestion logic but pointing to where we know the data is
    raw_path = "data/01_raw/LLCP2022_10rows.xpt"
    if not os.path.exists(raw_path):
        print("Data not found!")
        return

    df = load_and_process_data(raw_path, "data/02_intermediate")
    if df is None or df.empty:
        print("Data loading failed or empty.")
        return

    # Prepare X, y
    target = 'CVDINFR4' # From ingestion output
    if target not in df.columns:
         print(f"Target {target} not in columns.")
         return

    X = df.drop(columns=[target])
    y = df[target]

    # Handle object columns if any (simple encoding for test)
    X = X.select_dtypes(include=[np.number])

    # 2. Initialize Model
    print("\nInitializing XGBoostScratch...")
    model = XGBoostScratch(n_estimators=2, learning_rate=0.1, max_depth=2)

    # 3. Fit
    print("Fitting model...")
    try:
        model.fit(X, y)
        print("Fit complete.")
    except Exception as e:
        print(f"Fit failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Predict
    print("\nPredicting...")
    try:
        preds = model.predict_proba(X)
        print(f"Predictions (first 5): {preds[:5]}")

        binary_preds = model.predict(X)
        print(f"Binary Predictions (first 5): {binary_preds[:5]}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    print("\n=== Test Successful ===")

if __name__ == "__main__":
    run_integration_test()
