import json
import os

notebook_path = r"C:/Users/OMAR/Documents/Visual Studio 2022/Heart-Disease-Prediction-ML/notebooks/02_Training_PyCaret.ipynb"

def inject_debug_code(path):
    if not os.path.exists(path):
        print("Notebook not found.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated = False
    
    # We want to inject BEFORE the tuning cell.
    # We look for the cell containing "tuned_xgb = tune_model"
    
    for i, cell in enumerate(data['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "tuned_xgb = tune_model" in source:
                print(f"Found tuning cell at index {i}. Injecting debug cell before it...")
                
                debug_source = [
                    "# ==========================================\n",
                    "# DEBUG: VERIFICACIÓN DEL MODELO Y SETUP\n",
                    "# ==========================================\n",
                    "try:\n",
                    "    print(f\"🧐 Verificando Objeto XGB: {type(xgb)}\")\n",
                    "    \n",
                    "    # Check Imputation Config\n",
                    "    imp_type = get_config('imputation_type')\n",
                    "    num_imp = get_config('numeric_imputation_model')\n",
                    "    \n",
                    "    print(f\"🔧 Tipo de Imputación Global: {imp_type}\")\n",
                    "    print(f\"🔧 Modelo de Imputación Numérica: {num_imp}\")\n",
                    "    \n",
                    "    if imp_type == 'iterative':\n",
                    "        print(f\"ℹ️ NOTA: 'iterative' suele usar LightGBM por defecto. Por eso ves logs de LightGBM.\")\n",
                    "        print(\"   Si quieres evitarlo, reinicia el kernel para aplicar 'knn'.\")\n",
                    "    elif 'KNN' in str(num_imp):\n",
                    "        print(\"✅ Confirmado: Usando KNN. No deberías ver logs de LightGBM.\")\n",
                    "        \n",
                    "except Exception as e:\n",
                    "    print(f\"⚠️ Error en chequeo de debug: {e}\")\n"
                ]
                
                new_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": debug_source
                }
                
                # Insert before the tuning cell
                data['cells'].insert(i, new_cell)
                updated = True
                break
    
    if updated:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Notebook updated with debug cell.")
    else:
        print("Could not find tuning cell to inject debug code.")

inject_debug_code(notebook_path)
