import pandas as pd
import pyreadstat
import traceback

input_file = r'data\01_raw\LLCP2022.XPT'
output_file = r'data\01_raw\LLCP2022_10rows.xpt'

try:
    print(f"Reading {input_file}...")
    df, meta = pyreadstat.read_xport(input_file, row_limit=600, encoding='latin1')
    
    print(f"Read {len(df)} rows.")
    
    # Original labels mapping: column_name -> label
    original_labels = meta.column_names_to_labels
    
    # Rename columns starting with _ to avoid XPT validation error
    # And create a new label mapping for the renamed columns
    rename_map = {}
    new_labels = {}
    
    for col in df.columns:
        # Remove leading underscore if present
        new_col_name = col.lstrip('_')
        
        if new_col_name != col:
            rename_map[col] = new_col_name
        
        # Preserve the label, associating it with the new column name
        # If the column had a label, use it.
        if col in original_labels:
            new_labels[new_col_name] = original_labels[col]
            
    if rename_map:
        print(f"Renaming {len(rename_map)} columns to satisfy XPT format...")
        df.rename(columns=rename_map, inplace=True)
    
    print(f"Saving to {output_file}...")
    # Pass the new_labels dictionary to column_labels
    # Use version 5 for better compatibility with pd.read_sas
    pyreadstat.write_xport(df, output_file, 
                           file_label=meta.file_label, 
                           table_name=meta.table_name, 
                           file_format_version=5,
                           column_labels=new_labels)
    
    print("Done.")
    
    # Verification
    print("Verifying output file with pyreadstat...")
    df_verify, meta_verify = pyreadstat.read_xport(output_file)
    print(f"pyreadstat verification successful. Output file has {len(df_verify)} rows.")
    
    print("Verifying output file with pandas.read_sas...")
    try:
        df_pd = pd.read_sas(output_file, format='xport')
        print(f"pandas.read_sas verification successful. Shape: {df_pd.shape}")
    except Exception as e:
        print(f"pandas.read_sas failed: {e}")

    print("Checking for labels in new file...")
    # Check a few specific known labels
    test_cols = ['STATE', 'SEXVAR', 'GENHLTH'] # These likely existed (STATE was _STATE)
    found_labels = 0
    for col in test_cols:
        if col in meta_verify.column_names_to_labels:
            print(f"  {col}: {meta_verify.column_names_to_labels[col]}")
            found_labels += 1
        else:
            print(f"  {col}: <No Label Found>")
            
    if found_labels > 0:
        print("Labels appear to be preserved.")
    else:
        print("WARNING: No labels found for test columns.")

except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()
