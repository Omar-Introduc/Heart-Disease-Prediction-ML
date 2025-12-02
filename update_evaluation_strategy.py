import json

notebook_path = 'notebooks/03_Model_Evaluation.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

# Update the "Generate Predictions" section to use the custom threshold
# Look for where predictions are generated
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "predict_model" in "".join(cell['source']):
        new_source = [
            "# ==========================================\n",
            "# 2. GENERATE PREDICTIONS WITH CUSTOM THRESHOLD\n",
            "# ==========================================\n",
            "import json\n",
            "\n",
            "# Cargar umbral óptimo\n",
            "try:\n",
            "    with open('../models/threshold_config.json', 'r') as f:\n",
            "        thresh_config = json.load(f)\n",
            "    CUSTOM_THRESHOLD = thresh_config.get('optimal_threshold', 0.5)\n",
            "    print(f\"🔹 Usando Umbral Personalizado: {CUSTOM_THRESHOLD:.4f}\")\n",
            "except FileNotFoundError:\n",
            "    CUSTOM_THRESHOLD = 0.5\n",
            "    print(\"⚠️ No se encontró threshold_config.json, usando 0.5 por defecto\")\n",
            "\n",
            "# Generar probabilidades crudas\n",
            "predictions = predict_model(pipeline, data=df_eval, raw_score=True)\n",
            "\n",
            "# Recalcular etiquetas basadas en el umbral\n",
            "# Asumimos que prediction_score_1 es la prob de la clase positiva\n",
            "score_col = 'prediction_score_1' if 'prediction_score_1' in predictions.columns else 'prediction_score'\n",
            "\n",
            "# Si es 'prediction_score' genérico, ajustamos\n",
            "if score_col == 'prediction_score':\n",
            "    # Esto es tricky si las etiquetas ya vienen con 0.5. \n",
            "    # Mejor confiar en que PyCaret con raw_score=True da score_1 o similar.\n",
            "    # Si no, reconstruimos:\n",
            "    probs = predictions.apply(lambda x: x['prediction_score'] if x['prediction_label'] == 1 else 1 - x['prediction_score'], axis=1)\n",
            "else:\n",
            "    probs = predictions[score_col]\n",
            "\n",
            "# Aplicar nuevo umbral\n",
            "predictions['prediction_label'] = (probs >= CUSTOM_THRESHOLD).astype(int)\n",
            "predictions['prediction_score'] = probs # Unificamos para gráficos\n",
            "\n",
            "print(predictions[['prediction_label', 'prediction_score']].head())\n"
        ]
        cell['source'] = new_source
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook 03_Model_Evaluation.ipynb updated successfully.")
