import numpy as np
import pandas as pd
import pyreadstat
import os

def load_and_process_data(filepath, output_dir):
    """
    Loads SAS XPT file, cleans data, and saves as Parquet.
    """
    print(f"Loading data from {filepath}...")
    try:
        df, meta = pyreadstat.read_xport(filepath)
    except Exception as e:
        print(f"Error reading XPT file: {e}")
        return

    print(f"Initial shape: {df.shape}")

    # Semantic Leakage Removal (Issue 8.5)
    # Removing variables that indicate diagnosis or treatment consequence
    # Based on user advice and common BRFSS structure.
    # Since I couldn't find exact matches for 'aspirin' or 'diet' in the small snippet HTML grep,
    # I will attempt to remove common ones if they exist, or log warning.

    leakage_vars = [
        'CVDASPRN', # Aspirin usage (often asked to heart patients)
        'ASPUNSAF', # Aspirin unsafe
        'DIABEDU',  # Diabetes education (implies diagnosis)
        # Add others if found
    ]

    cols_to_drop = [col for col in leakage_vars if col in df.columns]
    if cols_to_drop:
        print(f"Dropping leakage variables: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    else:
        print("No predefined leakage variables found in this dataset (might be a subset).")

    # Basic cleaning
    # Convert object columns to category if suitable, or handle numeric codes.
    # The user warned: "Cuidado con columnas numéricas que son categorías codificadas".
    # Without the full codebook mapping, it's hard to know which is which automatically.
    # For this sprint, we will keep them as is but formatted for Parquet.

    # Save to intermediate
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "processed_data.parquet")
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path, index=False)
    print("Done.")

    return df

def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    Splits data into Train and Test.
    """
    # Simple random split for now
    # Ideally should use stratified split if target is imbalanced

    # Check if target exists
    if target_col not in df.columns:
        print(f"Target column {target_col} not found.")
        return None, None, None, None

    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Example usage
    raw_path = "data/01_raw/LLCP2022_10rows.xpt"
    if os.path.exists(raw_path):
        df = load_and_process_data(raw_path, "data/02_intermediate")

        # Determine target. 'CVDCRHD4' is Coronary Heart Disease.
        # 1 = Yes, 2 = No, 7 = Don't know, 9 = Refused.
        # We need to clean target: 1 -> 1, 2 -> 0, others -> drop or NaN?
        # For simplicity, let's treat 1 as positive, 2 as negative.

        target = 'CVDCRHD4'
        if target in df.columns:
            # Simple binary mapping for the example
            df = df[df[target].isin([1, 2])]
            df[target] = df[target].replace({2: 0})

            X_train, y_train, X_test, y_test = split_data(df, target)
            print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        else:
            print(f"Target {target} not found for split example.")
