import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os
import sys

def impute_clinical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements clinical imputation logic:
    1. Friedewald Formula for LDL.
    2. IterativeImputer for remaining physiological variables.
    3. Sanity Checks.
    """
    print("Starting Clinical Imputation...")
    df_imputed = df.copy()

    # Identify columns
    # We need to handle potential Spanish names from carga.py
    # or English names if already renamed.

    # 1. Friedewald Imputation
    # We check if we have the necessary columns.
    # We iterate rows or use vectorized operations.
    # Vectorized is better.

    # Helper to find column name
    def find_col(aliases):
        for a in aliases:
            if a in df_imputed.columns:
                return a
        return None

    col_tc = find_col(['Colesterol_Total', 'TotalCholesterol', 'LBXTC'])
    col_hdl = find_col(['HDL', 'Colesterol_HDL', 'LBDHDD'])
    col_trig = find_col(['Trigliceridos', 'Triglycerides', 'LBXTR'])
    col_ldl = find_col(['LDL', 'LBDLDL'])

    if col_ldl and col_tc and col_hdl and col_trig:
        print("Applying Friedewald Formula for missing LDL...")
        # Mask where LDL is null but others are not
        mask = (df_imputed[col_ldl].isnull()) & \
               (df_imputed[col_tc].notnull()) & \
               (df_imputed[col_hdl].notnull()) & \
               (df_imputed[col_trig].notnull())

        df_imputed.loc[mask, col_ldl] = df_imputed.loc[mask, col_tc] - \
                                        df_imputed.loc[mask, col_hdl] - \
                                        (df_imputed.loc[mask, col_trig] / 5.0)
        print(f"Filled {mask.sum()} missing LDL values using Friedewald.")
    else:
        print("Skipping Friedewald: Missing required columns (Total, HDL, Trig, LDL).")
        if not col_hdl:
            print("Note: HDL column is missing.")

    # 2. Iterative Imputer
    # Select numeric columns only for imputation
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude IDs or Target if necessary?
    # Usually we want to use all correlations. Target might help imputation but beware leakage.
    # If this is training data, using target is sometimes debated but often helpful.
    # However, usually we should separate X and y.
    # But for 'data_ingestion', we treat the dataset as a whole.
    # We should exclude 'SEQNO' or 'SEQN' if present.
    exclude_cols = [c for c in ['SEQN', 'SEQNO', 'id'] if c in numeric_cols]
    cols_to_impute = [c for c in numeric_cols if c not in exclude_cols]

    if df_imputed[cols_to_impute].isnull().sum().sum() > 0:
        print(f"Running IterativeImputer on {len(cols_to_impute)} columns...")
        # Constraints: Most clinical values are positive.
        imputer = IterativeImputer(min_value=0, max_value=np.inf, random_state=42, max_iter=10)
        df_imputed[cols_to_impute] = imputer.fit_transform(df_imputed[cols_to_impute])
    else:
        print("No missing values found for IterativeImputer.")

    # 3. Sanity Checks
    # LDL > TotalCholesterol is impossible (LDL is a part of Total)
    if col_ldl and col_tc:
        print("Performing Sanity Check: LDL <= TotalCholesterol...")
        # Check violations
        violations = df_imputed[df_imputed[col_ldl] > df_imputed[col_tc]]
        if not violations.empty:
            print(f"Found {len(violations)} rows where LDL > TotalCholesterol. Removing them.")
            df_imputed = df_imputed[df_imputed[col_ldl] <= df_imputed[col_tc]]
        else:
            print("Sanity Check Passed.")

    return df_imputed

def load_and_process_data(filepath: str, output_dir: str):
    """
    Loads data, applies clinical imputation, and saves to Parquet.
    """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return

    print(f"Loading data from {filepath}...")
    try:
        if filepath.endswith('.xpt'):
            df, _ = pd.read_sas(filepath, format='xport', encoding='latin1')
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            print("Unknown file format.")
            return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Initial shape: {df.shape}")

    # Apply Clinical Imputation
    df_clean = impute_clinical_data(df)

    print(f"Final shape: {df_clean.shape}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "process_data.parquet")
    print(f"Saving to {output_path}...")
    df_clean.to_parquet(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    # Default paths - adaptable based on where the raw/intermediate data is
    # Using the output of carga.py if available, or raw
    input_path = "data/02_processed/NHANES_ULTIMATE_CLEAN.parquet"
    output_dir = "data/02_intermediate"

    if os.path.exists(input_path):
        load_and_process_data(input_path, output_dir)
    else:
        print(f"Input {input_path} not found. Please ensure data ingestion (carga.py) has run.")
