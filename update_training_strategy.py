import json

notebook_path = 'notebooks/02_Training_PyCaret.ipynb'

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

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

# 1. Update scale_pos_weight logic (Force Bruta)
# Find the cell that calculates scale_pos_weight
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "scale_pos_weight =" in "".join(cell['source']):
        # Replace the calculation with a more aggressive one
        new_source = [
            "# ==========================================\n",
            "# 3. SELECCIÓN DE MODELO (Model Selection) - ESTRATEGIA IMBALANCE (FORCE BRUTA)\n",
            "# ==========================================\n",
            "# Calculamos el ratio de desbalance para usarlo en XGBoost\n",
            "from pycaret.classification import get_config\n",
            "y_train = get_config('y_train')\n",
            "neg_count = (y_train == 0).sum()\n",
            "pos_count = (y_train == 1).sum()\n",
            "\n",
            "# FORCE BRUTA: Aumentamos el peso de la clase positiva significativamente\n",
            "# Multiplicamos el ratio natural por 2 para penalizar más los falsos negativos\n",
            "scale_pos_weight = (neg_count / pos_count) * 2\n",
            "print(f\"Natural Ratio: {neg_count / pos_count:.2f}\")\n",
            "print(f\"Aggressive scale_pos_weight (Used): {scale_pos_weight:.2f}\")\n",
            "\n",
            "# En lugar de comparar todos, forzamos XGBoost con el peso corregido\n",
            "print(\"Creating XGBoost with aggressive scale_pos_weight...\")\n",
            "best_model = create_model('xgboost', scale_pos_weight=scale_pos_weight)\n",
            "\n",
            "# Mantenemos la variable top_models como lista para compatibilidad con celdas siguientes\n",
            "top_models = [best_model]\n",
            "print(f\"Top Model Selected: {best_model}\")\n"
        ]
        cell['source'] = new_source
        break

# 2. Add Threshold Moving Section
# We'll insert this BEFORE the "Finalize Model" section (usually near the end)
# Let's look for "## 6. Finalización del Modelo" or similar
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and "Finalización del Modelo" in "".join(cell['source']):
        insert_idx = i
        break

if insert_idx == -1:
    insert_idx = len(nb['cells']) - 1

threshold_cells = [
    create_markdown_cell([
        "## 5.2 Optimización de Umbral (Threshold Moving)\n",
        "Por defecto, el modelo decide con un umbral de 0.5. Sin embargo, en medicina preferimos **bajar el umbral** (ej. 0.2) para detectar más enfermos (Recall), aunque aumenten las falsas alarmas (Precision).\n",
        "\n",
        "Aquí buscamos el umbral óptimo que maximice el Recall sin destruir la Precisión."
    ]),
    create_code_cell([
        "from sklearn.metrics import precision_recall_curve\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import json\n",
        "\n",
        "# Predecir probabilidades en el set de validación (Hold-out de PyCaret)\n",
        "pred_holdout = predict_model(best_model, raw_score=True)\n",
        "y_true = pred_holdout[get_config('target')]\n",
        "\n",
        "# Obtener probabilidad de clase 1\n",
        "# PyCaret a veces llama a la columna 'prediction_score_1' o 'Score_1'\n",
        "score_col = 'prediction_score_1' if 'prediction_score_1' in pred_holdout.columns else 'prediction_score'\n",
        "# Si es 'prediction_score', hay que ver el label. Asumimos que si label=1 es score, si 0 es 1-score\n",
        "if score_col == 'prediction_score':\n",
        "    y_scores = pred_holdout.apply(lambda x: x['prediction_score'] if x['prediction_label'] == 1 else 1 - x['prediction_score'], axis=1)\n",
        "else:\n",
        "    y_scores = pred_holdout[score_col]\n",
        "\n",
        "# Calcular Precision-Recall para todos los umbrales\n",
        "precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)\n",
        "\n",
        "# Buscar el umbral que nos de un Recall >= 0.90 (o el máximo posible si no llega)\n",
        "target_recall = 0.90\n",
        "optimal_idx = np.argmax(recalls >= target_recall)\n",
        "# Si ningún punto cumple, tomamos el que tenga mejor F1\n",
        "if optimal_idx == 0 and recalls[0] < target_recall:\n",
        "    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)\n",
        "    optimal_idx = np.argmax(f1_scores)\n",
        "\n",
        "optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.2\n",
        "\n",
        "print(f\"✅ Umbral Óptimo Sugerido: {optimal_threshold:.4f}\")\n",
        "print(f\"   Recall esperado: {recalls[optimal_idx]:.4f}\")\n",
        "print(f\"   Precision esperada: {precisions[optimal_idx]:.4f}\")\n",
        "\n",
        "# Guardar el umbral para usarlo en Evaluación y Producción\n",
        "config_data = {'optimal_threshold': float(optimal_threshold)}\n",
        "with open('../models/threshold_config.json', 'w') as f:\n",
        "    json.dump(config_data, f)\n",
        "print(\"Umbral guardado en models/threshold_config.json\")\n",
        "\n",
        "# Visualizar Trade-off\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.plot(thresholds, precisions[:-1], \"b--\", label=\"Precision\")\n",
        "plt.plot(thresholds, recalls[:-1], \"g-\", label=\"Recall\")\n",
        "plt.axvline(x=optimal_threshold, color='r', linestyle=':', label=f'Optimal: {optimal_threshold:.2f}')\n",
        "plt.xlabel(\"Threshold\")\n",
        "plt.legend(loc=\"center left\")\n",
        "plt.title(\"Precision-Recall vs Threshold\")\n",
        "plt.show()"
    ])
]

# Insert the threshold cells
for cell in reversed(threshold_cells):
    nb['cells'].insert(insert_idx, cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook 02_Training_PyCaret.ipynb updated successfully.")
