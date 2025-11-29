import pandas as pd
import os

path = "data/02_intermediate/process_data.parquet"
if os.path.exists(path):
    try:
        df = pd.read_parquet(path)
        print("Dataset loaded successfully.")
        print(f"Shape: {df.shape}")
        print("Columns:")
        print(df.columns.tolist())
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print(f"File not found at {path}. Cannot extract headers.")
