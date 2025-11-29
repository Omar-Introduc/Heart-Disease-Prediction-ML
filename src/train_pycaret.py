import pandas as pd
import numpy as np
from pycaret.classification import setup, compare_models, save_model, pull, tune_model, predict_model, get_config, finalize_model, load_model
from sklearn.metrics import fbeta_score, recall_score, precision_score
import os
import json
from typing import Optional, List

def train_baseline(data_path: str, output_dir: str, target_col: str = 'CVDINFR4') -> None:
    """
    Trains baseline models using PyCaret with Tuning and Threshold Optimization.

    Args:
        data_path (str): Path to the input Parquet file.
        output_dir (str): Directory where the trained model and config will be saved.
        target_col (str): The name of the target variable column. Defaults to 'CVDINFR4'.
    """
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Clean indices immediately to avoid SMOTE crash
    df = df.reset_index(drop=True)

    # Check if target exists
    if target_col not in df.columns:
        # Fallback to CVDCRHD4 if specified target is missing but CVDCRHD4 is there
        if 'CVDCRHD4' in df.columns:
             print(f"Target {target_col} not found. Switching to CVDCRHD4.")
             target_col = 'CVDCRHD4'
        else:
            print(f"Target column {target_col} not found in dataset.")
            return

    print(f"Data shape: {df.shape}")

    # Sample data if too large to avoid MemoryError in Sandbox
    MAX_ROWS = 20000
    if len(df) > MAX_ROWS:
        print(f"Dataset too large ({len(df)} rows). Sampling {MAX_ROWS} rows for testing purposes...")
        # Maintain class distribution roughly
        df = df.sample(n=MAX_ROWS, random_state=42)
        print(f"New Data shape: {df.shape}")
        # Reset index again after sampling
        df = df.reset_index(drop=True)

    print(f"Target distribution:\n{df[target_col].value_counts()}")

    # PyCaret Setup
    print("Setting up PyCaret experiment...")
    # fix_imbalance=True uses SMOTE by default on train set
    exp = setup(
        data=df,
        target=target_col,
        session_id=42,
        fix_imbalance=True,
        verbose=False,
        html=False
    )

    # Compare Models
    # Issue 30: Force ensemble models
    print("Comparing models (sorted by Recall)...")
    best_model = compare_models(
        include=['xgboost', 'lightgbm', 'gbc', 'rf'],
        sort='Recall',
        n_select=1
    )

    print(f"Best model selected: {best_model}")

    # Tuning
    # Issue 31: Tuning for Recall
    print("Optimizing hyperparameters for Recall...")
    try:
        tuned_model = tune_model(best_model, optimize='Recall', n_iter=50, verbose=False)
        print("Tuning complete.")
    except Exception as e:
        print(f"Tuning failed or skipped: {e}")
        tuned_model = best_model

    results = pull()
    print("Tuning Results:")
    print(results.head())

    # Issue 33: Comparative Analysis and Optimal Threshold
    print("Calculating Optimal Threshold for F2-Score...")

    # Predict on holdout set
    predictions = predict_model(tuned_model, raw_score=True, verbose=False)

    # Identify target and score columns
    y_true = predictions[target_col]

    # Logic to find probability column
    score_col = None
    if 'prediction_score_1' in predictions.columns:
        score_col = 'prediction_score_1'
        y_scores = predictions['prediction_score_1']
    elif 'Score_1' in predictions.columns:
        score_col = 'Score_1'
        y_scores = predictions['Score_1']
    elif 'prediction_score' in predictions.columns:
        # Assuming binary classification where 1 is the positive class
        pred_label = predictions['prediction_label']
        raw_score = predictions['prediction_score']
        y_scores = np.where(pred_label == 1, raw_score, 1 - raw_score)
        score_col = 'inferred_score'
    else:
        print("Could not identify probability score column. Skipping threshold optimization.")
        y_scores = None

    if y_scores is not None:
        thresholds = np.arange(0, 1, 0.01)
        f2_scores = []
        recalls = []

        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)
            f2 = fbeta_score(y_true, y_pred, beta=2)
            f2_scores.append(f2)
            recalls.append(recall_score(y_true, y_pred))

        best_idx = np.argmax(f2_scores)
        best_thresh = thresholds[best_idx]
        best_f2 = f2_scores[best_idx]
        best_recall = recalls[best_idx]

        print(f"\n=== Optimal Threshold Analysis ===")
        print(f"Best Threshold (F2-Score): {best_thresh:.2f}")
        print(f"F2-Score at {best_thresh:.2f}: {best_f2:.4f}")
        print(f"Recall at {best_thresh:.2f}: {best_recall:.4f}")

        # Also print metrics at default 0.5 for comparison
        default_pred = (y_scores >= 0.5).astype(int)
        print(f"Metrics at 0.50 threshold:")
        print(f"F2-Score: {fbeta_score(y_true, default_pred, beta=2):.4f}")
        print(f"Recall: {recall_score(y_true, default_pred):.4f}")
        print("==================================\n")

    # Issue 34: Finalize Model (Train on Full Dataset)
    print("Finalizing model (training on full dataset)...")
    final_model = finalize_model(tuned_model)

    # Issue 35: Persistence
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save configuration
    config_data = {
        "threshold": float(best_thresh) if y_scores is not None else 0.5,
        "model_name": "Boosting"
    }
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    print(f"Model configuration saved to {config_path}")

    # Save finalized model
    model_name = "final_pipeline_v1"
    save_model(final_model, os.path.join(output_dir, model_name))
    print(f"Finalized model saved to {os.path.join(output_dir, model_name)}.pkl")

    # Issue 36: Final Report
    print("\n" + "="*50)
    print("FINAL TRAINING REPORT")
    print("="*50)
    print(f"Final Model: {final_model}")
    print(f"Optimal Threshold (F2): {config_data['threshold']:.4f}")
    if y_scores is not None:
        print(f"Best F2-Score: {best_f2:.4f}")
        print(f"Recall at Best Threshold: {best_recall:.4f}")
    print("-" * 30)
    print(f"Artifacts Saved in '{output_dir}/':")
    print(f"1. Model Pipeline: {model_name}.pkl")
    print(f"2. Config Metadata: model_config.json")
    print("="*50 + "\n")

if __name__ == "__main__":
    data_path = "data/02_intermediate/processed_data.parquet"
    output_dir = "models"
    train_baseline(data_path, output_dir)
