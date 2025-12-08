import json
import os

notebook_path = r"C:/Users/OMAR/Documents/Visual Studio 2022/Heart-Disease-Prediction-ML/notebooks/02_Training_PyCaret.ipynb"

def update_model_logic(path):
    if not os.path.exists(path):
        print("Notebook not found.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated_calibration = False
    
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Update Calibration to Sigmoid
            if "calibrate_model" in source and "isotonic" in source:
                print("Found isotonic calibration. Switching to sigmoid...")
                new_source = []
                for line in cell['source']:
                    if "method='isotonic'" in line or 'method="isotonic"' in line:
                         new_source.append(line.replace("isotonic", "sigmoid") + " # Updated to Sigmoid to prevent overfitting (Audit)\n")
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                updated_calibration = True
                
            # 2. Check Threshold Optimization (Often in a separate cell, but likely using 'calibrated_xgb')
            # If the user code creates 'final_model', we want to make sure it uses the RIGHT model.
            # Assuming 'calibrated_xgb' is the variable name.
            
    if updated_calibration:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Notebook updated: Calibration switched to Sigmoid.")
    else:
        print("Could not find 'isotonic' calibration to update. Checking file content manually might be needed.")

update_model_logic(notebook_path)
