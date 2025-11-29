import argparse
import pandas as pd
import numpy as np
import os
from pycaret.classification import load_model, predict_model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

def validate_external(model_path, data_path, output_path="external_validation_results.csv"):
    """
    Validates a PyCaret model on an external dataset.
    """
    print(f"Loading model from {model_path}...")
    # load_model appends .pkl automatically if not present, but let's handle it
    if model_path.endswith('.pkl'):
        model_name = model_path[:-4]
    else:
        model_name = model_path

    try:
        pipeline = load_model(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading data from {data_path}...")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        print("Unsupported file format. Please use .parquet or .csv")
        return

    # Standardize Columns (Mapping from Spanish to English as in train_pycaret.py)
    # This ensures compatibility if the external data uses the project's internal Spanish names
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

    # Apply renaming if columns exist
    df = df.rename(columns=rename_map)

    # Identify Target
    target_col = "HeartDisease"
    possible_targets = ["CVDINFR4", "Target", "Outcome", "HeartDisease", "TARGET"]
    found_target = False
    for t in possible_targets:
        if t in df.columns:
            target_col = t
            found_target = True
            break

    if not found_target:
        print("Warning: No target column found. Metrics cannot be calculated. Only predictions will be saved.")
    else:
        print(f"Target column identified: {target_col}")

    print("Generating predictions...")
    try:
        predictions = predict_model(pipeline, data=df)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    # Extract Prediction Label and Score
    # PyCaret usually adds 'prediction_label' and 'prediction_score'
    pred_label_col = "prediction_label"
    if pred_label_col not in predictions.columns:
         cols = [c for c in predictions.columns if 'label' in c]
         if cols: pred_label_col = cols[0]

    if found_target:
        y_true = predictions[target_col]
        y_pred = predictions[pred_label_col]

        # Determine score column for AUC
        # Newer PyCaret versions might output prediction_score_1

        y_score = None
        for col in predictions.columns:
             if 'score_1' in col or 'score_True' in col:
                 y_score = predictions[col]
                 break

        # If strictly prediction_score, convert based on label
        if y_score is None and 'prediction_score' in predictions.columns:
            # Assume binary 0/1
            # If label=1, score is prob(1). If label=0, score is prob(0) => prob(1) = 1 - score
            # We assume label 1 is the positive class.
            y_score = predictions.apply(lambda row: row['prediction_score'] if row[pred_label_col] == 1 else 1 - row['prediction_score'], axis=1)

        print("\n=== External Validation Metrics ===")
        try:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1 Score:  {f1:.4f}")

            if y_score is not None:
                try:
                    auc = roc_auc_score(y_true, y_score)
                    print(f"AUC:       {auc:.4f}")
                except Exception as e:
                    print(f"AUC:       Could not calculate ({e})")
        except Exception as e:
            print(f"Error calculating metrics: {e}")

    print(f"Saving predictions to {output_path}...")
    predictions.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External Validation Script")
    parser.add_argument("--model_path", required=True, help="Path to the trained PyCaret pipeline (.pkl)")
    parser.add_argument("--data_path", required=True, help="Path to the external dataset (.csv or .parquet)")
    parser.add_argument("--output_path", default="external_predictions.csv", help="Path to save predictions")

    args = parser.parse_args()

    validate_external(args.model_path, args.data_path, args.output_path)
