import json
import os
import re

notebook_path = r"C:/Users/OMAR/Documents/Visual Studio 2022/Heart-Disease-Prediction-ML/notebooks/02_Training_PyCaret.ipynb"

def update_notebook_setup(path):
    if not os.path.exists(path):
        print("Notebook not found.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    setup_updated = False
    
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "exp = setup(" in source:
                print("Found setup cell.")
                
                # We need to replace specific arguments in the setup call.
                # Since parsing python with regex is fragile, we'll do line-by-line replacement 
                # for the known keys inside the cell source.
                
                new_source = []
                for line in cell['source']:
                    # 1. Imputation
                    if "imputation_type=" in line or "imputation_type =" in line:
                        # Comment out old, add new
                        new_source.append(f"    # {line.strip()} # Disabled by Audit\n")
                        new_source.append("    imputation_type='simple', # Usamos simple para aplicar KNN abajo\n")
                        continue
                        
                    if "numeric_imputation=" in line or "numeric_imputation =" in line:
                        new_source.append(f"    # {line.strip()} # Disabled by Audit\n")
                        new_source.append("    numeric_imputation='knn', # Mejora para datos clinicos\n")
                        continue

                    # 2. Multicollinearity
                    if "multicollinearity_threshold=" in line or "multicollinearity_threshold =" in line:
                        new_source.append(f"    # {line.strip()} # Updated by Audit\n")
                        new_source.append("    multicollinearity_threshold=0.95, # Subido para no borrar features medicos importantes\n")
                        continue
                        
                    # 3. Feature Selection - Add it if missing, modify if present
                    if "setup(" in line:
                        new_source.append(line)
                        # Inject feature_selection right after setup( start or anywhere safe
                        # But purely appending args to the list is safer.
                        continue
                    
                    # Check if feature_selection is already there
                    if "feature_selection=" in line:
                        continue # We will add it definitively later or replace it
                        
                    new_source.append(line)
                
                # Insert key arguments if they weren't replaced (e.g. if previous config didn't have them explicit)
                # We iterate backwards to find the closing parenthesis ')' to insert before it
                # But finding the right ')' is hard in list of strings.
                # Easier strategy: Just rewrite the known kwargs in the `setup(...)` call by checking common patterns.
                
                # Let's try a safer replace approach on the joined string, then split back.
                full_cell_text = "".join(new_source)
                
                # Add feature_selection=True if not present
                if "feature_selection=" not in full_cell_text:
                    # Insert before the closing parenthesis of setup
                    # Assuming setup ends with ")\n" or similar.
                    # We will look for the last argument comma or just add it.
                    # Finding the end of setup call is tricky without AST.
                    
                    # Alternative: Setup usually ends args with , or none. 
                    # Let's add specific args to the list of params.
                    # We can treat the whole setup(...) call as a text block replacement
                    # but indentation matters. 
                    
                    # Hack: Replace `verbose=True` with `verbose=True, feature_selection=True` 
                    # as verbose is almost always there at the end.
                    if "verbose=True" in full_cell_text:
                        full_cell_text = full_cell_text.replace("verbose=True", "verbose=True,\n    feature_selection=True")
                
                cell['source'] = [s for s in full_cell_text.splitlines(keepends=True)]
                setup_updated = True
                print("Setup parameters updated.")
                break
    
    if setup_updated:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Notebook saved.")
    else:
        print("Could not find or update setup() cell.")

update_notebook_setup(notebook_path)
