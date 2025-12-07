import json
import os

notebook_path = r"C:/Users/OMAR/Documents/Visual Studio 2022/Heart-Disease-Prediction-ML/notebooks/02_Training_PyCaret.ipynb"

def inject_imputation_settings(path):
    if not os.path.exists(path):
        print("Notebook not found.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    setup_updated = False
    
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            source_list = cell['source']
            source_text = "".join(source_list)
            
            if "exp = setup(" in source_text:
                print("Found setup cell for injection.")
                
                new_source = []
                # Check if we already have imputation settings (from previous attempts or manual)
                has_imputation = any("numeric_imputation" in line for line in source_list)
                
                for line in source_list:
                    # We inject right after 'setup(' line or 'data=df,'
                    new_source.append(line)
                    
                    if "exp = setup(" in line and not has_imputation:
                        new_source.append("    imputation_type='simple', # Usamos simple para aplicar KNN abajo\n")
                        new_source.append("    numeric_imputation='knn', # Mejora para datos clinicos (Audit)\n")
                        has_imputation = True # Prevent double insertion per cell
                
                cell['source'] = new_source
                setup_updated = True
                print("Imputation parameters injected.")
                break
    
    if setup_updated:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Notebook saved with injection.")
    else:
        print("Could not find setup() or imputation already present.")

inject_imputation_settings(notebook_path)
