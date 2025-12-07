import json
import os

notebook_path = r"C:/Users/OMAR/Documents/Visual Studio 2022/Heart-Disease-Prediction-ML/notebooks/03_Model_Evaluation.ipynb"

def get_refined_robust_code():
    return [
        "# Helper to extract feature importance from various model types (Linear, Tree, Calibrated)\n",
        "def get_model_importance(model):\n",
        "    try:\n",
        "        # Case 1: Tree-based\n",
        "        if hasattr(model, 'feature_importances_'):\n",
        "            return model.feature_importances_\n",
        "        # Case 2: Linear models (use absolute coef)\n",
        "        if hasattr(model, 'coef_'):\n",
        "            return np.abs(model.coef_[0])\n",
        "        # Case 3: Wrappers (CalibratedClassifier, etc.)\n",
        "        if hasattr(model, 'estimator'):\n",
        "            return get_model_importance(model.estimator)\n",
        "        if hasattr(model, 'base_estimator'):\n",
        "            return get_model_importance(model.base_estimator)\n",
        "        if hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:\n",
        "             # Average over calibrated classifiers. Note: They usually wrap the estimator in 'estimator' attribute\n",
        "             imps = []\n",
        "             for clf in model.calibrated_classifiers_:\n",
        "                 if hasattr(clf, 'estimator'):\n",
        "                     imps.append(get_model_importance(clf.estimator))\n",
        "                 elif hasattr(clf, 'base_estimator'):\n",
        "                     imps.append(get_model_importance(clf.base_estimator))\n",
        "             if imps:\n",
        "                 return np.mean(np.vstack(imps), axis=0)\n",
        "    except Exception as e:\n",
        "        print(f\"Debug extraction error: {e}\")\n",
        "        pass\n",
        "    return None\n",
        "\n",
        "# Previous helper\n",
        "def get_base_estimator(estimator):\n",
        "    if hasattr(estimator, 'steps'):\n",
        "        return estimator.steps[-1][1]\n",
        "    return estimator\n",
        "\n"
    ]

if os.path.exists(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Find the cell to fix (fi_weights_code)
    target_cell = None
    for cell in data['cells']:
        if cell.get('id') == 'fi_weights_code':
            target_cell = cell
            break
    
    if target_cell:
        print("Updating cell logic with REFINED helper...")
        # Completely replace the source with the refined robust logic
        target_cell['source'] = get_refined_robust_code() + [
            "# Extraer y visualizar Feature Importances del modelo cargado\n",
            "try:\n",
            "    print(\"📊 Generando gráfico de pesos de variables...\")\n",
            "    \n",
            "    if 'pipeline' in locals():\n",
            "        # 1. Get final estimator from pipeline\n",
            "        model_step = pipeline.steps[-1][1]\n",
            "        final_model = get_base_estimator(model_step)\n",
            "        \n",
            "        # 2. Get Importances Robustly\n",
            "        importances = get_model_importance(final_model)\n",
            "\n",
            "        if importances is not None:\n",
            "            # 3. Get Feature Names\n",
            "            feature_names = []\n",
            "            if hasattr(final_model, 'feature_names_in_'):\n",
            "                feature_names = final_model.feature_names_in_\n",
            "            elif hasattr(pipeline[:-1], 'get_feature_names_out'):\n",
            "                 try:\n",
            "                     feature_names = pipeline[:-1].get_feature_names_out()\n",
            "                 except:\n",
            "                     pass\n",
            "            \n",
            "            # 4. Plot\n",
            "            if len(feature_names) != len(importances):\n",
            "                feature_names = [f'Feature {i}' for i in range(len(importances))]\n",
            "            \n",
            "            s_imp = pd.Series(importances, index=feature_names)\n",
            "            \n",
            "            plt.figure(figsize=(10, 8))\n",
            "            s_imp.nlargest(20).sort_values().plot(kind='barh', color='skyblue', edgecolor='black')\n",
            "            plt.title(\"Top 20 Feature Importances (Model Weights)\")\n",
            "            plt.xlabel(\"Importance Score\")\n",
            "            plt.grid(axis='x', linestyle='--', alpha=0.7)\n",
            "            plt.tight_layout()\n",
            "            plt.show()\n",
            "        else:\n",
            "            print(f\"⚠️ No se pudieron extraer importancias del modelo: {type(final_model)}\")\n",
            "    else:\n",
            "        print(\"⚠️ Validar que la variable 'pipeline' esté cargada correctamente.\")\n",
            "        \n",
            "except Exception as e:\n",
            "    print(f\"❌ Error generando el gráfico de importancia: {e}\")"
        ]
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Notebook updated with REFINED robust logic.")
            
    else:
        print("Target cell not found! Please run the previous fix first to append it.")

else:
    print("Notebook not found.")
