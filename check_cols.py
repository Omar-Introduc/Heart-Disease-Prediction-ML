import pandas as pd
import os

path = "data/02_intermediate/processed_data.parquet"
if os.path.exists(path):
    df = pd.read_parquet(path)
    print("Columns:", df.columns.tolist()[:20]) # First 20
    print("Columns with underscore:", [c for c in df.columns if c.startswith('_')])
else:
    print("File not found")
