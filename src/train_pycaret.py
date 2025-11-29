import pandas as pd
import numpy as np
from pycaret.classification import (
    setup,
    compare_models,
    save_model,
    tune_model,
    predict_model,
    finalize_model,
)
from sklearn.metrics import fbeta_score
import os
import json


def train_baseline(
    data_path: str, output_dir: str, target_col: str = "HeartDisease"
) -> None:
    """
    Trains baseline models using PyCaret on NHANES Clinical Data.

    Args:
        data_path: Path to the processed NHANES parquet file.
        output_dir: Output directory for models.
        target_col: Name of the target variable.
    """
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    df = df.reset_index(drop=True)

    # Check target and Rename columns to English (NHANES Standard)
    rename_map = {
        "TARGET": "HeartDisease",
        "Edad": "Age",
        "Sexo": "Sex",
        "Raza": "Race",
        "Educacion": "Education",
        "Presion_Sistolica": "SystolicBP",
        "Cintura": "WaistCircumference",
        "Altura": "Height",
        "Colesterol_Total": "TotalCholesterol",
        "Trigliceridos": "Triglycerides",
        "Glucosa": "Glucose",
        "Creatinina": "Creatinine",
        "Acido_Urico": "UricAcid",
        "Fumador": "Smoking",
        "Actividad_Fisica": "PhysicalActivity",
        "Seguro_Medico": "HealthInsurance",
    }
    df = df.rename(columns=rename_map)

    if target_col not in df.columns:
        # Fallback to likely targets if default name is wrong
        possible_targets = ["CVDINFR4", "Target", "Outcome", "HeartDisease", "TARGET"]
        found = False
        for t in possible_targets:
            if t in df.columns:
                target_col = t
                print(f"Target found: {target_col}")
                found = True
                break
        if not found:
            print(f"Target column {target_col} not found.")
            return

    # Sample if too large for sandbox
    MAX_ROWS = 20000
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

    print(f"Target distribution:\n{df[target_col].value_counts()}")

    # Define Feature Types
    # Numeric Features: All continuous clinical variables
    numeric_features = [
        "Age",
        "BMI",
        "SystolicBP",
        "DiastolicBP",
        "TotalCholesterol",
        "LDL",
        "Triglycerides",
        "HbA1c",
        "Glucose",
        "UricAcid",
        "Creatinine",
        "WaistCircumference",
        "Height",
    ]
    # Categorical Features: Binary/Nominal
    categorical_features = [
        "Sex",
        "Smoking",
        "Alcohol",
        "PhysicalActivity",
        "HealthInsurance",
    ]

    # Filter columns that exist in df
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    print("Setting up PyCaret...")
    # normalize=True is CRITICAL for mixed scales (Age vs Cholesterol)
    setup(
        data=df,
        target=target_col,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        normalize=True,
        normalize_method="minmax",  # or zscore
        session_id=42,
        fix_imbalance=True,
        verbose=False,
    )

    print("Comparing models...")
    best_model = compare_models(sort="Recall", n_select=1)
    print(f"Best model: {best_model}")

    print("Tuning...")
    try:
        tuned_model = tune_model(
            best_model, optimize="Recall", n_iter=20, verbose=False
        )
    except Exception:
        tuned_model = best_model

    # Threshold Optimization
    print("Optimizing Threshold...")
    predictions = predict_model(tuned_model, raw_score=True, verbose=False)
    y_true = predictions[target_col]

    # Locate score column
    y_scores = None
    for col in predictions.columns:
        if "score" in col.lower() and ("1" in col or "True" in str(col)):
            y_scores = predictions[col]
            break
    if y_scores is None:
        # If binary 0/1, maybe infer?
        if "prediction_score" in predictions.columns:
            # Assuming label is prediction_label
            label = predictions["prediction_label"]
            score = predictions["prediction_score"]
            y_scores = np.where(label == 1, score, 1 - score)

    best_thresh = 0.5
    if y_scores is not None:
        thresholds = np.arange(0.1, 0.9, 0.01)
        f2_scores = []
        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)
            f2_scores.append(fbeta_score(y_true, y_pred, beta=2))
        best_idx = np.argmax(f2_scores)
        best_thresh = thresholds[best_idx]
        print(f"Best Threshold (F2): {best_thresh:.2f}")

    print("Finalizing model...")
    final_model = finalize_model(tuned_model)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_model(final_model, os.path.join(output_dir, "final_pipeline_v1"))

    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump({"threshold": float(best_thresh)}, f)

    print("Done.")


if __name__ == "__main__":
    # Updated path
    # Using the file found in data/02_intermediate
    data_path = "data/02_intermediate/process_data.parquet"
    output_dir = "models"
    train_baseline(data_path, output_dir)
