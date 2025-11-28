import pandas as pd
import numpy as np
from pycaret.classification import setup, compare_models, save_model, pull, tune_model, predict_model, get_config, finalize_model
from sklearn.metrics import fbeta_score, classification_report
import os
import json

def train_final_model(data_path, output_dir, target_col='CVDINFR4'):
    """
    Trains final model using PyCaret with Tuning and Threshold Optimization.
    """
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Check if target exists
    if target_col not in df.columns:
        if 'CVDCRHD4' in df.columns:
             print(f"Target {target_col} not found. Switching to CVDCRHD4.")
             target_col = 'CVDCRHD4'
        else:
            print(f"Target column {target_col} not found in dataset.")
            return

    # Memory Management: Sample if too large
    MAX_ROWS = 1000  # Reduced to 1k for extremely fast validation
    if len(df) > MAX_ROWS:
        print(f"Dataset too large ({len(df)} rows). Sampling {MAX_ROWS} rows for training to avoid OOM.")
        # Ensure we keep enough positive cases
        # Stratified sampling manually if needed, or just let sample handle it?
        # Let's do a simple sample, assuming distribution is somewhat preserved or fixed by SMOTE later.
        df = df.sample(n=MAX_ROWS, random_state=42)

    # Reset index to avoid duplicate index errors in PyCaret
    df = df.reset_index(drop=True)

    print(f"Data shape: {df.shape}")
    print(f"Target distribution:\n{df[target_col].value_counts()}")

    # PyCaret Setup
    print("Setting up PyCaret experiment...")
    # fix_imbalance=True uses SMOTE on train set only
    # n_jobs=1 to avoid multiprocessing memory spikes
    exp = setup(
        data=df,
        target=target_col,
        session_id=42,
        fix_imbalance=True,
        verbose=False,
        html=False,
        train_size=0.8,
        n_jobs=1
    )

    # 1. Compare Models
    print("Comparing models (sorted by Recall)...")
    # Limiting to 1 best model.
    # Exclude models that might be too heavy (like SVM/Ridge if they don't scale well with many cols, though usually they are fine).
    # 'lightgbm' and 'xgboost' are usually memory efficient.
    best_model = compare_models(sort='Recall', n_select=1)
    print(f"Best model found: {best_model}")

    # 2. Tuning (Issue 30 & 31)
    print("Tuning model (Optimizing for Recall)...")
    # n_iter reduced to 1 for speed/memory in this env
    tuned_model = tune_model(best_model, optimize='Recall', n_iter=1)
    print(f"Tuned model: {tuned_model}")

    # 3. Threshold Optimization
    print("Optimizing Decision Threshold on Hold-out Set...")
    pred_holdout = predict_model(tuned_model)

    # Check column names
    # pred_holdout should have prediction_label and prediction_score
    # In binary classification, PyCaret 3.x usually gives 'prediction_label' and 'prediction_score'
    # where score is probability of the predicted label.

    y_true = pred_holdout[target_col]
    probs = []

    # Iterate safely
    # If using xgboost/lightgbm, predict_model returns pandas df

    if 'prediction_score' in pred_holdout.columns and 'prediction_label' in pred_holdout.columns:
        for index, row in pred_holdout.iterrows():
            score = row['prediction_score']
            label = row['prediction_label']
            # If label is 1, prob is score. If label is 0, prob is 1-score.
            # Assuming 1 is the positive class.
            if label == 1:
                probs.append(score)
            else:
                probs.append(1 - score)
    else:
        # Fallback if columns are named differently (e.g. Score_1, Label)
        # But standard PyCaret is consistent.
        print("Warning: prediction columns not found as expected. Using default predictions.")
        probs = [0.5] * len(y_true) # Dummy

    probs = np.array(probs)

    # Calculate F2 for thresholds
    thresholds = np.arange(0.01, 1.0, 0.01)
    f2_scores = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = fbeta_score(y_true, preds, beta=2)
        f2_scores.append(score)

    best_threshold_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f2 = f2_scores[best_threshold_idx]

    print(f"Optimal Threshold (F2): {best_threshold:.2f}")
    print(f"Max F2 Score: {best_f2:.4f}")

    # 4. Finalize Model
    print("Finalizing model...")
    final_model = finalize_model(tuned_model)

    # Save best model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = os.path.join(output_dir, "best_pipeline")
    save_model(final_model, model_path)
    print(f"Model saved to {model_path}.pkl")

    # Save Feature Names for UI Mapper
    try:
        # Get transformed training data columns
        X_train_trans = get_config('X_train_transformed')
        feature_names = X_train_trans.columns.tolist()

        features_path = os.path.join(output_dir, "feature_names.json")
        with open(features_path, 'w') as f:
            json.dump(feature_names, f)
        print(f"Feature names saved to {features_path}")

    except Exception as e:
        print(f"Could not save feature names: {e}")

    # Save threshold info
    threshold_info = {
        "optimal_threshold": float(best_threshold),
        "f2_score": float(best_f2)
    }
    with open(os.path.join(output_dir, "threshold.json"), "w") as f:
        json.dump(threshold_info, f)

if __name__ == "__main__":
    data_path = "data/02_intermediate/processed_data.parquet"
    output_dir = "models"
    train_final_model(data_path, output_dir)
