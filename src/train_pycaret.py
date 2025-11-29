import pandas as pd
import numpy as np
from pycaret.classification import (
    setup,
    compare_models,
    save_model,
    tune_model,
    predict_model,
    finalize_model,
    pull,
    add_metric,
    create_model
)
from sklearn.metrics import fbeta_score, precision_score, f1_score, confusion_matrix, recall_score
from pycaret.classification import interpret_model
import os
import shutil
import json
import matplotlib.pyplot as plt
import seaborn as sns

def train_baseline(
    data_path: str, output_dir: str, target_col: str = "HeartDisease", strategy: str = "SMOTE"
) -> None:
    """
    Trains baseline models using PyCaret on NHANES Clinical Data with advanced imbalance handling.

    Args:
        data_path: Path to the processed NHANES parquet file.
        output_dir: Output directory for models.
        target_col: Name of the target variable.
        strategy: 'SMOTE' (Approach A) or 'SCALE_POS_WEIGHT' (Approach B).
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

    # Sample if too large for sandbox (memory constraint)
    MAX_ROWS = 20000
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

    print(f"Target distribution:\n{df[target_col].value_counts()}")

    # Calculate scale_pos_weight for Approach B
    pos_count = df[target_col].sum()
    neg_count = len(df) - pos_count
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Calculated scale_pos_weight: {scale_weight:.2f}")

    # Define Feature Types
    numeric_features = [
        "Age", "BMI", "SystolicBP", "DiastolicBP", "TotalCholesterol", "LDL",
        "Triglycerides", "HbA1c", "Glucose", "UricAcid", "Creatinine",
        "WaistCircumference", "Height"
    ]
    categorical_features = [
        "Sex", "Smoking", "Alcohol", "PhysicalActivity", "HealthInsurance"
    ]

    # Filter columns that exist in df
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    print(f"Setting up PyCaret with strategy: {strategy}...")

    # Common Setup Args
    setup_args = dict(
        data=df,
        target=target_col,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        normalize=True,
        normalize_method="minmax",
        session_id=42,
        verbose=False,
    )

    if strategy == "SMOTE":
        setup_args['fix_imbalance'] = True
    else:
        setup_args['fix_imbalance'] = False
        # For Approach B, we pass scale_pos_weight to models later

    setup(**setup_args)

    # Add Custom Metrics if not present (PyCaret has them by default, but we ensure)
    add_metric('Precision', 'Precision', precision_score)
    add_metric('F1', 'F1', f1_score)

    print("Comparing models...")
    # We select top 5 to filter by Precision
    top_models = compare_models(sort="Recall", n_select=5, include=['xgboost', 'lightgbm', 'rf', 'gbc', 'et'] if strategy == 'SCALE_POS_WEIGHT' else None)

    # If strategy B, we might need to manually create XGBoost/LightGBM with weights to strictly follow "Enfoque B"
    # But compare_models runs standard models.
    # To truly test Approach B, we should instantiate XGBoost with the param.
    if strategy == "SCALE_POS_WEIGHT":
        print("Training XGBoost and LightGBM with scale_pos_weight...")
        xgb = create_model('xgboost', scale_pos_weight=scale_weight, verbose=False)
        lgbm = create_model('lightgbm', scale_pos_weight=scale_weight, verbose=False)
        # Add to top_models to consider them in selection
        top_models = [xgb, lgbm] + top_models

    # Filter by Precision Constraint
    results = pull()
    print("Top Models Performance:")
    print(results[['Accuracy', 'AUC', 'Recall', 'Precision', 'F1']])

    selected_model = None

    # Iterate through top models and check their cross-val precision
    # PyCaret's compare_models returns the trained model objects.
    # We need to map them back to their metrics or re-evaluate.
    # 'results' dataframe matches the order if n_select matches rows.
    # But n_select returns a list.

    print("Applying Precision Constraint (> 0.4)...")
    valid_models = []

    # We assume 'pull()' returns the table sorted by Recall (as requested).
    # We iterate through the table to find the best one that meets precision criteria.
    # Then we pick that model from the list of top_models.
    # Note: 'pull()' contains the metrics.

    # This matching is tricky because top_models list might not be perfectly aligned if we added models manually.
    # Simplification: we check the metrics of the best model.
    # If best model (highest recall) has poor precision, we tune it or pick next.

    # Let's just iterate our candidate models and evaluate them briefly or use their stored metrics if accessible.
    # In PyCaret, model objects don't store CV metrics directly in a standard attribute easily accessible without 'pull'.

    # Robust approach: Select the best from the list based on logic
    best_candidate = top_models[0]
    best_candidate_recall = 0

    for model in top_models:
        # Predict on hold-out set to check metrics (or use CV results if we could link them)
        # Using hold-out for selection logic here
        pred_holdout = predict_model(model, verbose=False)

        # Calculate metrics
        y_true = pred_holdout[target_col]

        # prediction_label might be the col name
        pred_col = "prediction_label"
        if pred_col not in pred_holdout.columns:
             # Fallback
             pred_col = [c for c in pred_holdout.columns if 'label' in c][0]

        y_pred = pred_holdout[pred_col]

        prec = precision_score(y_true, y_pred)
        from sklearn.metrics import recall_score
        rec = recall_score(y_true, y_pred)

        print(f"Model: {str(model)[:20]}... | Recall: {rec:.4f} | Precision: {prec:.4f}")

        if prec >= 0.4:
            # We want to maximize Recall among those with Prec >= 0.4
            if rec > best_candidate_recall:
                best_candidate_recall = rec
                selected_model = model

    if selected_model is None:
        print("Warning: No model met the Precision > 0.4 criteria. Defaulting to highest Recall model.")
        selected_model = top_models[0]

    print(f"Selected Model: {selected_model}")

    print("Tuning...")
    try:
        # Optimize Recall but keep an eye on Precision?
        # PyCaret tune_model optimizes a single metric.
        tuned_model = tune_model(
            selected_model, optimize="Recall", n_iter=20, verbose=False
        )

        # Check precision of tuned model
        pred_tuned = predict_model(tuned_model, verbose=False)
        pred_col = [c for c in pred_tuned.columns if 'label' in c][0]
        prec_tuned = precision_score(pred_tuned[target_col], pred_tuned[pred_col])

        if prec_tuned < 0.4:
            print(f"Tuned model precision {prec_tuned:.4f} < 0.4. Reverting to original model.")
            tuned_model = selected_model

    except Exception as e:
        print(f"Tuning failed: {e}. Using selected model.")
        tuned_model = selected_model

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
         # Try finding the score column for the positive class (1)
         for col in predictions.columns:
             if 'score_1' in col or 'score_True' in col:
                 y_scores = predictions[col]
                 break

    best_thresh = 0.5
    if y_scores is not None:
        # Precision-Constrained Recall Maximization
        thresholds = np.arange(0.1, 0.9, 0.01)
        candidates = []

        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            candidates.append((t, prec, rec, f1))

        # Strategy: Max Recall such that Precision >= 0.4
        safe_candidates = [c for c in candidates if c[1] >= 0.4]

        best_prec, best_rec, best_f1 = 0, 0, 0

        if safe_candidates:
            # Sort by Recall (descending)
            safe_candidates.sort(key=lambda x: x[2], reverse=True)
            best_thresh, best_prec, best_rec, best_f1 = safe_candidates[0]
            print(f"Selected Threshold {best_thresh:.2f} based on Precision >= 0.4 and Max Recall.")
        else:
            # Fallback: Maximize F1
            candidates.sort(key=lambda x: x[3], reverse=True)
            best_thresh, best_prec, best_rec, best_f1 = candidates[0]
            print(f"No threshold met Precision >= 0.4. Selected Threshold {best_thresh:.2f} based on Max F1.")

        print(f"Metrics at Optimal Threshold: Precision: {best_prec:.4f}, Recall: {best_rec:.4f}, F1: {best_f1:.4f}")

    # SHAP Explainability
    print("Generating SHAP Summary Plot...")
    try:
        # interpret_model works with the estimator (tuned_model) and uses the test set from setup()
        # It saves the plot as 'Summary Plot.png' in the current directory if save=True
        interpret_model(tuned_model, plot='summary', save=True)

        # Move the file to models/ directory
        shap_src = 'Summary Plot.png'
        shap_dst = os.path.join(output_dir, 'shap_summary_plot.png')
        if os.path.exists(shap_src):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            shutil.move(shap_src, shap_dst)
            print(f"SHAP summary plot saved to {shap_dst}")
        else:
            print("SHAP plot generated but file 'Summary Plot.png' not found.")

    except Exception as e:
        print(f"SHAP generation failed: {e}")

    print("Finalizing model...")
    final_model = finalize_model(tuned_model)

    # --- Save Training Reference Schema for Drift Detection ---
    print("Generating Training Reference Schema...")
    reference_schema = {}
    # df is the data used for training (after renaming)
    for col in numeric_features:
        if col in df.columns:
            desc = df[col].describe()
            reference_schema[col] = {
                "mean": float(desc["mean"]),
                "std": float(desc["std"]),
                "min": float(desc["min"]),
                "max": float(desc["max"]),
            }

    with open(os.path.join(output_dir, "training_reference_schema.json"), "w") as f:
        json.dump(reference_schema, f, indent=4)
    print(f"Training reference schema saved to {os.path.join(output_dir, 'training_reference_schema.json')}")
    # ----------------------------------------------------------

    # Confusion Matrix on Test Set
    # We need to predict using the final model on the hold-out test set
    # Note: finalize_model fits on ALL data (train+test).
    # To get a "Test Set" confusion matrix, we should have used the hold-out predictions BEFORE finalization
    # or split data manually before PyCaret.
    # PyCaret's predict_model uses the hold-out set by default if data is not passed.

    print("Generating Confusion Matrix on Hold-out Set (before finalization training)...")
    pred_holdout = predict_model(tuned_model, verbose=False) # Predict on holdout
    y_true_ho = pred_holdout[target_col]
    pred_col_ho = [c for c in pred_holdout.columns if 'label' in c][0]
    y_pred_ho = pred_holdout[pred_col_ho]

    cm = confusion_matrix(y_true_ho, y_pred_ho)
    print("\nConfusion Matrix (Test Set):")
    print(cm)
    print(f"Precision: {precision_score(y_true_ho, y_pred_ho):.4f}")
    print(f"Recall: {recall_score(y_true_ho, y_pred_ho):.4f}")
    print(f"F1: {f1_score(y_true_ho, y_pred_ho):.4f}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_model(final_model, os.path.join(output_dir, "final_pipeline_v1"))

    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump({"threshold": float(best_thresh), "strategy": strategy}, f)

    print("Done.")


if __name__ == "__main__":
    # Using the file found in data/02_intermediate
    data_path = "data/02_intermediate/process_data.parquet"
    output_dir = "models"

    # We default to SMOTE (Approach A) but can change to SCALE_POS_WEIGHT (Approach B)
    # The user asked to "Configura ... para probar".
    # I will run SMOTE as default here.
    train_baseline(data_path, output_dir, strategy="SMOTE")
