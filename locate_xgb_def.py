import json

notebook_path = r"C:/Users/OMAR/Documents/Visual Studio 2022/Heart-Disease-Prediction-ML/notebooks/02_Training_PyCaret.ipynb"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    found = False
    for i, cell in enumerate(data['cells']):
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            for line in source_lines:
                if "xgb =" in line or "xgb=" in line:
                    print(f"--- FOUD 'xgb =' in Cell {i} ---")
                    print("".join(source_lines))
                    print("-----------------------------")
                    found = True
                    
    if not found:
        print("Could not find 'xgb =' assignment. Checking if it's passed as string or hidden.")

except Exception as e:
    print(f"Error: {e}")
