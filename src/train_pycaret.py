import pandas as pd
from pycaret.classification import setup, compare_models, save_model, pull
import os

def train_baseline(data_path, output_dir, target_col='CVDINFR4'):
    """
    Trains baseline models using PyCaret.
    """
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

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
    print(f"Target distribution:\n{df[target_col].value_counts()}")

    # PyCaret Setup
    # fix_imbalance=True uses SMOTE by default on train set
    # session_id for reproducibility
    print("Setting up PyCaret experiment...")
    exp = setup(
        data=df,
        target=target_col,
        session_id=42,
        fix_imbalance=True,
        verbose=False,
        html=False # Disable HTML for script execution
    )

    # Compare Models
    # Sort by Recall as requested
    print("Comparing models (sorted by Recall)...")
    best_model = compare_models(sort='Recall', n_select=1)

    results = pull()
    print("Top models:")
    print(results.head())

    # Save best model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = os.path.join(output_dir, "best_pipeline")
    save_model(best_model, model_path)
    print(f"Model saved to {model_path}.pkl")

if __name__ == "__main__":
    data_path = "data/02_intermediate/processed_data.parquet"
    output_dir = "models"
    train_baseline(data_path, output_dir)
