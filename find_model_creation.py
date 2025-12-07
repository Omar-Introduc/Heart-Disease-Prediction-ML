import json

notebook_path = r"C:/Users/OMAR/Documents/Visual Studio 2022/Heart-Disease-Prediction-ML/notebooks/02_Training_PyCaret.ipynb"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i, cell in enumerate(data['cells']):
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            for j, line in enumerate(source_lines):
                if "create_model" in line:
                    print(f"--- Found create_model in Cell {i}, Line {j} ---")
                    # Print context
                    start = max(0, j-2)
                    end = min(len(source_lines), j+5)
                    print("".join(source_lines[start:end]))
                    print("---------------------------------------------")

except Exception as e:
    print(f"Error: {e}")
