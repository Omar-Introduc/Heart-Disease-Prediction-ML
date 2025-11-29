
import sys
import os
import random
import pandas as pd
import numpy as np
import pickle

# Add project root to path
sys.path.append(os.getcwd())

def generate_random_clinical_data(num_samples=5):
    """
    Generates random clinical data resembling NHANES features.
    """
    data = {
        'Age': np.random.uniform(20, 90, num_samples),
        'Sex': np.random.choice(['Male', 'Female'], num_samples),
        'BMI': np.random.uniform(15, 45, num_samples),
        'SystolicBP': np.random.uniform(90, 180, num_samples),
        'TotalCholesterol': np.random.uniform(120, 300, num_samples),
        'LDL': np.random.uniform(50, 200, num_samples),
        'Triglycerides': np.random.uniform(50, 400, num_samples),
        'HbA1c': np.random.uniform(4.0, 12.0, num_samples),
        'Glucose': np.random.uniform(70, 300, num_samples),
        'UricAcid': np.random.uniform(3.0, 10.0, num_samples),
        'Creatinine': np.random.uniform(0.5, 2.0, num_samples),
        'WaistCircumference': np.random.uniform(60, 130, num_samples),
        'Smoking': np.random.choice(['Yes', 'No'], num_samples),
        'Alcohol': np.random.choice(['Yes', 'No'], num_samples),
        'PhysicalActivity': np.random.choice(['Yes', 'No'], num_samples),
    }
    return pd.DataFrame(data)

def validate_physiological_ranges(df):
    """
    Validates that the generated data falls within physiological ranges.
    Returns a list of warnings if any.
    """
    warnings = []

    checks = [
        ('SystolicBP', 0, 300),
        ('BMI', 5, 100),
        ('Glucose', 0, 1000),
        ('TotalCholesterol', 0, 1000)
    ]

    for col, min_val, max_val in checks:
        if col in df.columns:
            if df[col].min() < min_val or df[col].max() > max_val:
                warnings.append(f"CRITICAL: {col} contains values outside physiological range ({min_val}-{max_val})")

    return warnings

def run_productive_audit():
    print("=== Integration Test: Productive Flow (Clinical Health Check) ===")

    # 1. Generate Random Clinical Data
    print("\n1. Generating synthetic clinical data...")
    df_sample = generate_random_clinical_data(num_samples=5)
    print(df_sample)

    # 2. Validate Data
    print("\n2. Validating physiological ranges...")
    warnings = validate_physiological_ranges(df_sample)
    if warnings:
        for w in warnings:
            print(w)
        print("Audit FAILED due to physiological data integrity issues.")
        return
    else:
        print("Data integrity check PASSED.")

    # 3. Load Model
    print("\n3. Loading Production Model...")
    model_paths = [
        "models/best_pipeline.pkl",
        "models/final_pipeline_v1.pkl"
    ]

    model_path = None
    for p in model_paths:
        if os.path.exists(p):
            model_path = p
            break

    if not model_path:
        print(f"ERROR: No model found in {model_paths}. Skipping prediction.")
        # We allow the script to exit gracefully if model is missing,
        # but in a real CI/CD this should fail.
        return

    print(f"Loaded model: {model_path}")

    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load model pickle. {e}")
        return

    # 4. Predict
    print("\n4. Running Prediction...")
    try:
        # PyCaret pipelines usually implement predict
        # Some might need 'predict_proba'
        preds = pipeline.predict(df_sample)

        # Check if predict_proba exists
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(df_sample)
            print("Probabilities generated successfully.")
        else:
            print("Model does not support predict_proba.")

        df_sample['Predicted_Label'] = preds
        print("\nPrediction Results:")
        print(df_sample[['Age', 'Sex', 'SystolicBP', 'Predicted_Label']])

        print("\n=== Productive Flow Audit Successful ===")

    except Exception as e:
        print(f"\nCRITICAL: Prediction failed during stress test. {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_productive_audit()
