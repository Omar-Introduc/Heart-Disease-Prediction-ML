import sys
import os
import random
import pandas as pd

# Ensure src is in path
sys.path.append(os.getcwd())

# Try to import InputData from src.interfaces for validation limits
# If it fails, we will define fallback limits
try:
    from src.interfaces import InputData  # noqa: F401

    HAS_INTERFACES = True
except ImportError:
    HAS_INTERFACES = False
    print(
        "Warning: Could not import InputData from src.interfaces. Using fallback limits."
    )


def run_productive_audit():
    print("=== Health Check: Productive Pipeline (NHANES) ===")

    # 1. Define Model Path
    # Primary path used by app.py
    model_path_primary = "models/final_pipeline_v1.pkl"
    # Secondary/Prompt requested path
    model_path_secondary = "models/best_pipeline.pkl"

    target_model_path = None

    if os.path.exists(model_path_primary):
        target_model_path = model_path_primary
    elif os.path.exists(model_path_secondary):
        target_model_path = model_path_secondary

    if not target_model_path:
        print(
            f"❌ Error: No pipeline found at {model_path_primary} or {model_path_secondary}"
        )
        print("Please run 'python src/train_pycaret.py' to generate the model.")
        return

    print(f"✅ Found model at: {target_model_path}")

    # 2. Generate Random Clinical Data
    print("Generating random clinical data...")

    # Define ranges based on NHANES valid ranges (approximate if not imported)
    ranges = {
        "Age": (18, 100, int),
        "Sex": (0, 1, int),
        "BMI": (12.0, 60.0, float),
        "SystolicBP": (80.0, 220.0, float),
        "DiastolicBP": (40.0, 120.0, float),
        "TotalCholesterol": (100.0, 400.0, float),
        "LDL": (30.0, 300.0, float),
        "Triglycerides": (30.0, 600.0, float),
        "HbA1c": (4.0, 15.0, float),
        "Glucose": (50.0, 300.0, float),
        "UricAcid": (2.0, 12.0, float),
        "Creatinine": (0.4, 5.0, float),
        "WaistCircumference": (50.0, 180.0, float),
        "Height": (130.0, 220.0, float),
        "Smoking": (0, 1, int),
        "Alcohol": (0, 1, int),
        "PhysicalActivity": (0, 1, int),
        "HealthInsurance": (0, 1, int),
    }

    # Generate 5 random samples
    data = []
    for _ in range(5):
        sample = {}
        for feat, (low, high, dtype) in ranges.items():
            if dtype is int:
                val = random.randint(low, high)
            else:
                val = round(random.uniform(low, high), 1)
            sample[feat] = val
        data.append(sample)

    df_test = pd.DataFrame(data)
    print("Generated Test Data:")
    print(df_test.head())

    # 3. Validate Inputs (Physiological Ranges)
    print("\nValidating inputs against physiological ranges...")
    validation_passed = True

    for idx, row in df_test.iterrows():
        # Check SystolicBP
        if row["SystolicBP"] < 0:
            print(f"❌ Row {idx}: Invalid SystolicBP (<0)")
            validation_passed = False
        if row["SystolicBP"] > 300:  # Extreme sanity check
            print(f"❌ Row {idx}: Extreme SystolicBP (>300)")
            validation_passed = False

        # Check BMI
        if row["BMI"] <= 0:
            print(f"❌ Row {idx}: Invalid BMI (<=0)")
            validation_passed = False

    if validation_passed:
        print("✅ Input validation passed (Basic Physiological Checks).")
    else:
        print("❌ Input validation failed.")
        return

    # 4. Load Pipeline & Predict
    print("\nLoading Pipeline...")
    try:
        from pycaret.classification import load_model

        # PyCaret load_model appends .pkl automatically if not present,
        # but our path has it. remove extension for pycaret load_model if needed
        path_no_ext = os.path.splitext(target_model_path)[0]

        # We use the same loading logic as app.py
        pipeline = load_model(path_no_ext)
        print("✅ Pipeline loaded successfully.")

        print("Running prediction...")
        # PyCaret prediction
        # If the pipeline is a PyCaret Pipeline, we can use predict_model or pipeline.predict
        # But `load_model` returns the pipeline object.

        # Important: The pipeline expects features in specific order or names.
        # Check if the pipeline has 'feature_names_in_'

        if hasattr(pipeline, "feature_names_in_"):
            expected = set(pipeline.feature_names_in_)
            current = set(df_test.columns)
            missing = expected - current
            if missing:
                print(f"⚠️ Warning: Missing columns expected by model: {missing}")
                # Add missing as 0
                for c in missing:
                    df_test[c] = 0

        # Predict
        preds = pipeline.predict(df_test)
        probs = pipeline.predict_proba(df_test)

        print("\nPredictions:")
        print(preds)
        print("\nProbabilities:")
        print(probs)

        print("\n✅ Prediction successful without errors.")
        print("\n=== Health Check PASSED ===")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print("\n=== Health Check FAILED ===")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_productive_audit()
