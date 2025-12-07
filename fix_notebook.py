import json
import os

notebook_path = r"C:/Users/OMAR/Documents/Visual Studio 2022/Heart-Disease-Prediction-ML/notebooks/03_Model_Evaluation.ipynb"

def get_base_estimator_code():
    return [
        "# Define helper to get base estimator\n",
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
        # Check if function is already defined
        source = target_cell.get('source', [])
        is_defined = any("def get_base_estimator" in line for line in source)
        
        if not is_defined:
            print("Injecting get_base_estimator definition...")
            # Insert at the beginning
            target_cell['source'] = get_base_estimator_code() + source
            
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=1)
            print("Notebook updated successfully.")
        else:
            print("Function already defined.")
            
    else:
        print("Target cell 'fi_weights_code' not found. Appending it...")
        # Create the cell if it doesn't exist (copy from append_eval_cell.py logic but with fix)
        new_cells = [
          {
           "cell_type": "markdown",
           "id": "fi_weights_md",
           "metadata": {},
           "source": [
            "## 6. Importancia de Variables (Pesos del Modelo)\n",
            "\n",
            "A diferencia de SHAP (que explica el impacto en la salida), este gráfico muestra qué variables usa más internamente el modelo (ej. ganancia en árboles)."
           ]
          },
          {
           "cell_type": "code",
           "execution_count": None,
           "id": "fi_weights_code",
           "metadata": {},
           "outputs": [],
           "source": get_base_estimator_code() + [
            "# Extraer y visualizar Feature Importances del modelo cargado\n",
            "try:\n",
            "    print(\"📊 Generando gráfico de pesos de variables...\")\n",
            "    \n",
            "    # 1. Obtener el estimador final del pipeline\n",
            "    if 'pipeline' in locals():\n",
            "        model_step = pipeline.steps[-1][1]\n",
            "        final_model = get_base_estimator(model_step)\n",
            "\n",
            "        # 2. Intentar obtener importancias\n",
            "        if hasattr(final_model, 'feature_importances_'):\n",
            "            importances = final_model.feature_importances_\n",
            "            \n",
            "            feature_names = []\n",
            "            if hasattr(final_model, 'feature_names_in_'):\n",
            "                feature_names = final_model.feature_names_in_\n",
            "            elif hasattr(pipeline[:-1], 'get_feature_names_out'):\n",
            "                 try:\n",
            "                     feature_names = pipeline[:-1].get_feature_names_out()\n",
            "                 except:\n",
            "                     pass\n",
            "            \n",
            "            if len(feature_names) == len(importances):\n",
            "                s_imp = pd.Series(importances, index=feature_names)\n",
            "            else:\n",
            "                s_imp = pd.Series(importances, index=[f'Feature {i}' for i in range(len(importances))])\n",
            "            \n",
            "            plt.figure(figsize=(10, 8))\n",
            "            s_imp.nlargest(20).sort_values().plot(kind='barh', color='skyblue', edgecolor='black')\n",
            "            plt.title(\"Top 20 Feature Importances (Model Weights)\")\n",
            "            plt.xlabel(\"Importance Score\")\n",
            "            plt.grid(axis='x', linestyle='--', alpha=0.7)\n",
            "            plt.tight_layout()\n",
            "            plt.show()\n",
            "        else:\n",
            "            print(\"⚠️ El modelo final no expone el atributo 'feature_importances_'.\")\n",
            "    else:\n",
            "        print(\"⚠️ Validar que la variable 'pipeline' esté cargada correctamente.\")\n",
            "        \n",
            "except Exception as e:\n",
            "    print(f\"❌ Error generando el gráfico de importancia: {e}\")"
           ]
          }
        ]
        data['cells'].extend(new_cells)
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Notebook updated (appended new cells).")

else:
    print("Notebook file not found.")
