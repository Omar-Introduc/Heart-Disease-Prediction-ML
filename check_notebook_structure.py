import json
import os

notebook_path = r"C:/Users/OMAR/Documents/Visual Studio 2022/Heart-Disease-Prediction-ML/notebooks/03_Model_Evaluation.ipynb"

if os.path.exists(notebook_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total cells: {len(data['cells'])}")
        print("Cell IDs in order:")
        for i, cell in enumerate(data['cells']):
            print(f"{i}: {cell.get('id', 'NO_ID')} - Type: {cell['cell_type']}")
            if cell.get('id') == 'fi_weights_code':
                source_preview = "".join(cell['source'])[:100]
                print(f"   -> FOUND fi_weights_code! Source start: {source_preview}...")
                if "get_model_importance" in "".join(cell['source']):
                     print("   -> VERIFIED: Robust logic is present.")
                else:
                     print("   -> WARNING: Robust logic NOT found.")

    except Exception as e:
        print(f"Error reading notebook: {e}")
else:
    print("Notebook file not found.")
