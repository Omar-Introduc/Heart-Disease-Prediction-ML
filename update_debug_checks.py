import json
import os

notebook_path = r"C:/Users/OMAR/Documents/Visual Studio 2022/Heart-Disease-Prediction-ML/notebooks/02_Training_PyCaret.ipynb"

def update_debug_cell(path):
    if not os.path.exists(path):
        print("Notebook not found.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated = False
    
    for cell in data['cells']:
        source = "".join(cell['source'])
        if "DEBUG: VERIFICACIÓN DEL MODELO" in source:
            print("Found existing debug cell. Updating...")
            
            new_source = [
                "# ==========================================\n",
                "# DEBUG: VERIFICACIÓN DEL MODELO Y SETUP\n",
                "# ==========================================\n",
                "try:\n",
                "    from pycaret.classification import get_config\n",
                "    print(f\"🧐 Verificando Objeto XGB: {type(xgb)}\")\n",
                "    \n",
                "    print(\"🔍 Inspeccionando Pipeline (pasos de preprocesamiento)...\")\n",
                "    pp = get_config('pipeline')\n",
                "    \n",
                "    found_lgbm_source = False\n",
                "    for name, step in pp.steps:\n",
                "        print(f\"   Step: {name} -> {type(step).__name__}\")\n",
                "        \n",
                "        # Check Imputer\n",
                "        if 'Imputer' in str(type(step).__name__):\n",
                "             if hasattr(step, 'estimator'):\n",
                "                 print(f\"      Imputer Estimator: {step.estimator}\")\n",
                "             else:\n",
                "                 print(f\"      Imputer Strategy: {getattr(step, 'strategy', 'Unknown')}\")\n",
                "\n",
                "        # Check Feature Selection\n",
                "        if 'Select' in name or 'Selection' in str(type(step).__name__):\n",
                "             print(f\"      Selector detected!\")\n",
                "             if hasattr(step, 'estimator'):\n",
                "                 est = step.estimator\n",
                "                 print(f\"      Selector Estimator: {type(est)}\")\n",
                "                 if 'LGBM' in str(type(est)) or 'LightGBM' in str(type(est)):\n",
                "                     print(\"      🚨 CAUSA ENCONTRADA: 'feature_selection=True' usa LightGBM internamente.\")\n",
                "                     found_lgbm_source = True\n",
                "\n",
                "    if found_lgbm_source:\n",
                "        print(\"\\n✅ CONCLUSIÓN: Los logs de LightGBM son NORMALES. \")\n",
                "        print(\"   Vienen del proceso de selección de features (activado por auditoría).\")\n",
                "        print(\"   Tu modelo final SIGUE SIENDO XGBoost. Puedes continuar tranquilo.\")\n",
                "    else:\n",
                "        print(\"\\n❓ No se encontró LightGBM explícito en el pipeline. Revisar logs detallados.\")\n",
                "        \n",
                "except Exception as e:\n",
                "    print(f\"⚠️ Error en chequeo de debug: {e}\")\n"
            ]
            
            cell['source'] = new_source
            updated = True
            break
    
    if updated:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Notebook updated with IMPROVED debug cell.")
    else:
        print("Could not find debug cell to update.")

update_debug_cell(notebook_path)
