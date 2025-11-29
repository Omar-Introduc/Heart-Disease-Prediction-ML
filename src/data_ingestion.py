import pandas as pd
import pyreadstat
import os
from typing import Optional, Tuple


def load_and_process_data(filepath: str, output_dir: str) -> Optional[pd.DataFrame]:
    """
    Loads SAS XPT file, cleans data, and saves as Parquet.

    Args:
        filepath (str): Path to the input SAS XPT file.
        output_dir (str): Directory where the processed Parquet file will be saved.

    Returns:
        Optional[pd.DataFrame]: The processed dataframe, or None if loading fails.
    """
    print(f"Loading data from {filepath}...")
    try:
        df, meta = pyreadstat.read_xport(filepath, encoding="latin1")
    except Exception as e:
        print(f"Error reading XPT file: {e}")
        return None

    print(f"Initial shape: {df.shape}")

    # Clean column names (remove leading '_')
    df.columns = [col.lstrip("_") for col in df.columns]
    print("Cleaned column names (removed leading '_').")

    # Handle IDs
    if "SEQNO" in df.columns:
        print("Setting SEQNO as index.")
        df = df.set_index("SEQNO")

    # Semantic Leakage Removal
    # Removing variables that indicate diagnosis or treatment consequence
    leakage_vars = [
        "CVDASPRN",  # Aspirin usage
        "ASPUNSAF",  # Aspirin unsafe
        "DIABEDU",  # Diabetes education
        # Add others if found
    ]

    # Check for leakage vars in cleaned columns
    cols_to_drop = [col for col in leakage_vars if col in df.columns]
    if cols_to_drop:
        print(f"Dropping leakage variables: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Target Standardization
    # Priority: CVDINFR4 (Heart Attack) > CVDCRHD4 (Coronary Heart Disease)
    target = None
    if "CVDINFR4" in df.columns:
        target = "CVDINFR4"
    elif "CVDCRHD4" in df.columns:
        target = "CVDCRHD4"

    if target:
        print(f"Target variable selected: {target}")
        # Filter valid values: 1 (Yes), 2 (No)
        # 7 (Don't know), 9 (Refused) are treated as missing or dropped
        df = df[df[target].isin([1, 2])]

        # Map to 0/1: 1->1 (Yes), 2->0 (No)
        df[target] = df[target].replace({2: 0}).astype(int)

        # Rename target to 'Target' for consistency? Or keep original?
        # Keeping original for traceability, but printing count
        print(f"Target distribution:\n{df[target].value_counts()}")
    else:
        print("Warning: No suitable target variable (CVDINFR4 or CVDCRHD4) found.")

    # Save to intermediate
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "processed_data_profundo.parquet")
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path)  # Index (SEQNO) is preserved in Parquet
    print("Done.")

    return df


def split_data(
    df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits data into Train and Test.

    Args:
        df (pd.DataFrame): The full dataframe.
        target_col (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: X_train, y_train, X_test, y_test.
        Returns (None, None, None, None) if target column is missing.
    """
    if target_col not in df.columns:
        print(f"Target column {target_col} not found.")
        return None, None, None, None

    # Shuffle
    df = df.sample(frac=1, random_state=random_state)  # Index preserved

    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    raw_path = "data/01_raw/LLCP2022.xpt"
    # Fallback to empty if not found, just to test script syntax
    if os.path.exists(raw_path):
        load_and_process_data(raw_path, "data/02_intermediate")
    else:
        print(f"File {raw_path} not found.")
