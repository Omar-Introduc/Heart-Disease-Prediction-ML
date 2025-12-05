import json
import os

notebook_path = r"C:/Users/OMAR/Documents/Visual Studio 2022/Heart-Disease-Prediction-ML/notebooks/02_Training_PyCaret.ipynb"

new_cells = [
  {
   "cell_type": "markdown",
   "id": "feature_importance_md",
   "metadata": {},
   "source": [
    "## 6. Importancia de Variables (Feature Importance)\n",
    "\n",
    "Analizamos qué variables tienen mayor peso en la predicción del modelo."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "feature_importance_code",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualizar la importancia de las características\n",
    "try:\n",
    "    print(\"📊 Generando gráfico de importancia de variables...\")\n",
    "    plot_model(tuned_xgb, plot='feature')\n",
    "except Exception as e:\n",
    "    print(f\"⚠️ No se pudo generar el gráfico automáticamente: {e}\")\n",
    "    print(\"Intentando método alternativo si el modelo es accesible...\")\n",
    "    try:\n",
    "        # Método alternativo manual si plot_model falla\n",
    "        import matplotlib.pyplot as plt\n",
    "        import seaborn as sns\n",
    "        \n",
    "        # Acceder al modelo XGBoost interno\n",
    "        # PyCaret envuelve el modelo, a veces es necesario navegar la estructura\n",
    "        # Esto es un intento genérico\n",
    "        model_obj = tuned_xgb \n",
    "        if hasattr(tuned_xgb, 'steps'): # Si es pipeline\n",
    "            model_obj = tuned_xgb.steps[-1][1]\n",
    "            \n",
    "        if hasattr(model_obj, 'feature_importances_'):\n",
    "            # Intentar obtener nombres de columnas si es posible, sino usar índices\n",
    "            try:\n",
    "               cols = get_config('X_train').columns\n",
    "            except:\n",
    "               cols = range(len(model_obj.feature_importances_))\n",
    "               \n",
    "            feat_importances = pd.Series(model_obj.feature_importances_, index=cols)\n",
    "            plt.figure(figsize=(10, 8))\n",
    "            feat_importances.nlargest(15).plot(kind='barh')\n",
    "            plt.title('Top 15 Feature Importance')\n",
    "            plt.xlabel('Relative Importance')\n",
    "            plt.show()\n",
    "    except Exception as e2:\n",
    "        print(f\"❌ No se pudo generar el gráfico alternativo: {e2}\")"
   ]
  }
]

if os.path.exists(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if cell already exists to avoid duplicates
    existing_source = [c.get('source',[]) for c in data['cells']]
    is_present = False
    for src in existing_source:
        if src and "plot_model(tuned_xgb, plot='feature')" in "".join(src):
            is_present = True
            break
            
    if not is_present:
        data['cells'].extend(new_cells)
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Successfully appended cells.")
    else:
        print("Cells already present.")
else:
    print(f"Notebook not found at {notebook_path}")
