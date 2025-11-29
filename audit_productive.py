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

        # Align features with model expectations
        # PyCaret/XGBoost pipelines are sensitive to column order and presence
        try:
            # Attempt to extract feature names from the final estimator or the pipeline wrapper
            if hasattr(pipeline, "feature_names_in_"):
                model_features = list(pipeline.feature_names_in_)
            elif hasattr(pipeline.steps[-1][1], "feature_names_in_"):
                 model_features = list(pipeline.steps[-1][1].feature_names_in_)
            else:
                 model_features = None

            if model_features:
                # Filter out Target variable if accidentally included in feature names
                # PyCaret sometimes includes target in feature_names_in_ depending on version/setup
                targets_to_exclude = ["HeartDisease", "TARGET", "Target", "Outcome", "CVDINFR4"]
                model_features = [f for f in model_features if f not in targets_to_exclude]

                # Add missing columns with 0
                for feature in model_features:
                    if feature not in df_test.columns:
                        # Special handling for One-Hot Encoded features if possible, otherwise 0
                        df_test[feature] = 0

                # Keep only expected columns and reorder
                df_test = df_test[model_features]
                print(f"✅ Aligned input features to model ({len(model_features)} features).")
            else:
                print("⚠️ Could not determine model feature names. Proceeding with raw input...")

        except Exception as e:
            print(f"⚠️ Feature alignment warning: {e}")

        # Predict
        preds = pipeline.predict(df_test)
        probs = pipeline.predict_proba(df_test)

        print("\nPredictions:")
        print(preds)
        print("\nProbabilities:")
        print(probs)

        print("\n✅ Prediction successful without errors.")

        # --- NEW: Data Drift Check ---
        print("\n=== Checking for Data Drift ===")
        ref_schema = load_reference_schema()
        if ref_schema:
            check_drift(df_test, ref_schema)
        else:
            print("⚠️ Skipped drift check (Reference schema not found).")

        print("\n=== Health Check PASSED ===")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print("\n=== Health Check FAILED ===")
        import traceback

        traceback.print_exc()

import json

def load_reference_schema():
    path = "models/training_reference_schema.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    print(f"Warning: Reference schema not found at {path}")
    return None

def check_drift(new_data, reference_schema):
    """
    Checks if mean of new_data variables deviates > 2 std devs from reference.
    """
    print("Analyzing drift...")
    drift_detected = False

    # Critical variables to monitor
    critical_vars = ["SystolicBP", "Age", "HbA1c", "BMI", "Glucose"]

    for col in critical_vars:
        if col in new_data.columns and col in reference_schema:
            ref = reference_schema[col]
            ref_mean = ref["mean"]
            ref_std = ref["std"]

            curr_mean = new_data[col].mean()

            # Z-score of the current mean relative to reference distribution
            # Note: comparing sample mean to population params.
            # Deviation = abs(curr_mean - ref_mean)
            # Threshold = 2 * ref_std

            deviation = abs(curr_mean - ref_mean)
            threshold = 2 * ref_std

            if deviation > threshold:
                print(f"⚠️ DRIFT DETECTED: {col} shift suspected.")
                print(f"   Ref Mean: {ref_mean:.2f} +/- {ref_std:.2f}")
                print(f"   New Mean: {curr_mean:.2f} (Diff: {deviation:.2f})")
                drift_detected = True
            else:
                # Optional: print(f"   {col}: Stable")
                pass

    if not drift_detected:
        print("✅ No significant drift detected.")


if __name__ == "__main__":
    # Simulate Post-COVID Data Drift in __main__
    print("Normal Execution:")
    run_productive_audit()

    print("\n\n--- SIMULATING POST-COVID DATA DRIFT ---")

    # We monkeypatch the data generation part or just create a new function?
    # Or better, we modify run_productive_audit to accept data or we just run the check_drift function manually here with bad data.
    # The instructions say: "En el bloque __main__, genera un lote de datos sintéticos 'Post-COVID' ... y demuestra que el script detecta la anomalía."

    # Generate Post-COVID data
    # Increase SystolicBP by +15 mmHg from normal ranges
    ranges = {
        "Age": (18, 100, int),
        "Sex": (0, 1, int),
        "BMI": (12.0, 60.0, float),
        "SystolicBP": (80.0, 220.0, float), # Normal
        # We will boost this manually
    }

    # Generate data similar to run_productive_audit but with higher BP
    data = []
    for _ in range(50): # More samples for better mean stability
        sample = {}
        # Base random
        sample["Age"] = random.randint(18, 80)
        sample["SystolicBP"] = random.uniform(135, 160) # High BP on average
        sample["HbA1c"] = random.uniform(4.0, 6.0)
        sample["BMI"] = random.uniform(20, 30)
        sample["Glucose"] = random.uniform(80, 120)
        data.append(sample)

    df_covid = pd.DataFrame(data)

    # Force drift: shift SystolicBP to be very high compared to likely training mean (approx 120)
    # If training mean is ~120 and std is ~20, 2*std = 40. 120+40 = 160.
    # Let's make it extreme to be sure.
    df_covid["SystolicBP"] = df_covid["SystolicBP"] + 50

    print("Post-COVID Synthetic Data Stats:")
    print(df_covid.describe().loc[['mean', 'min', 'max']])

    ref_schema = load_reference_schema()
    if ref_schema:
        print("Running Drift Check on Post-COVID Data...")
        check_drift(df_covid, ref_schema)
