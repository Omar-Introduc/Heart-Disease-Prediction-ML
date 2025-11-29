
import json

def update_notebook(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)

    # 1. Update Header (Cell 0)
    header_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📈 Evaluación del Modelo Final (Métricas y Gráficos)\n",
            "\n",
            "## 🎯 Objetivo\n",
            "En este notebook evaluamos el rendimiento del modelo seleccionado (`best_pipeline.pkl`) utilizando datos no vistos (o un subconjunto de validación).\n",
            "Analizamos métricas clave para clasificación binaria en el contexto médico.\n",
            "\n",
            "## 📊 Métricas Principales\n",
            "- **Matriz de Confusión**: ¿Cuántos enfermos detectamos correctamente (TP) y cuántos sanos alarmamos falsamente (FP)?\n",
            "- **Recall (Sensibilidad)**: Capacidad del modelo para identificar positivos. Es nuestra prioridad.\n",
            "- **Precision**: De los que el modelo dice que están enfermos, ¿cuántos lo están realmente?\n",
            "- **F1-Score**: Balance armónico entre Precision y Recall.\n",
            "- **AUC-ROC**: Capacidad discriminante global del modelo.\n",
            "\n",
            "## 🔍 Interpretabilidad\n",
            "- **Feature Importance**: ¿Qué biomarcadores (Edad, Glucosa, Presión) influyen más en la predicción?"
        ]
    }

    new_cells = [header_cell]

    for cell in nb['cells']:
        source_text = "".join(cell['source'])

        if "MODEL_PATH =" in source_text:
             new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Configuración y Carga del Modelo\n",
                    "\n",
                    "Cargamos el pipeline entrenado y el dataset. Usamos un subconjunto (`frac=0.2`) para simular un set de validación rápida."
                ]
            })
             new_cells.append(cell)

        elif "# 1. LOAD MODEL" in source_text: # Skip this markdown logic if inserted above, actually the code block handles loading.
             # The original cell 1 has code for loading model. I put header before it above.
             # Let's verify. Original cell 0 imports. Cell 1 loads model.
             # My logic above was slightly off. Cell 0 is imports/config. Cell 1 is loading.
             pass

        elif "predict_model" in source_text:
             new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Generación de Predicciones\n",
                    "\n",
                    "Aplicamos el modelo a los datos de evaluación.\n",
                    "PyCaret añade automáticamente:\n",
                    "- `prediction_label`: La clase predicha (0 o 1).\n",
                    "- `prediction_score`: La probabilidad asociada a la predicción."
                ]
            })
             new_cells.append(cell)

        elif "confusion_matrix" in source_text and "ConfusionMatrixDisplay" in source_text:
             new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Matriz de Confusión\n",
                    "\n",
                    "Visualizamos los aciertos y errores:\n",
                    "- **Verdaderos Positivos (TP)**: Enfermos detectados correctamente (Cuadrante inferior derecho).\n",
                    "- **Falsos Negativos (FN)**: Enfermos no detectados (Cuadrante inferior izquierdo). **¡Es el error más peligroso!**"
                ]
            })
             new_cells.append(cell)

        elif "classification_report" in source_text:
             new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Reporte de Métricas Detallado\n",
                    "\n",
                    "Generamos un reporte con Precision, Recall y F1-Score para ambas clases.\n",
                    "También calculamos el **AUC (Area Under the Curve)** para medir la calidad global del clasificador (1.0 es perfecto, 0.5 es aleatorio)."
                ]
            })
             new_cells.append(cell)

        elif "plot_model" in source_text or "feature_importances_" in source_text:
             new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Importancia de Variables (Feature Importance)\n",
                    "\n",
                    "¿Qué variables pesan más en la decisión del modelo?\n",
                    "En modelos médicos, esperamos ver variables como Edad, Presión Sistólica o Glucosa en el top. Esto valida clínicamente el modelo."
                ]
            })
             new_cells.append(cell)
        else:
             # Add imports cell (Cell 0 original) back if missed
             if "import pandas" in source_text:
                 new_cells.append(cell)
             # Catch-all for others
             elif cell not in new_cells: # Avoid duplicates if logic was imperfect
                 # This check is weak because dict comparison issues.
                 # Let's fix logic:
                 # Cell 0 (Imports) -> Handled by "if 'import pandas'" above? Yes.
                 # Cell 1 (Load) -> Handled by "if 'MODEL_PATH ='" check? No, that check matches cell 0 in original.
                 # Wait, let's look at original notebook content again.
                 # Cell 0: Imports + Config
                 # Cell 1: Load Model & Data
                 # Cell 2: Predict
                 # Cell 3: Confusion Matrix
                 # Cell 4: Metrics
                 # Cell 5: Feature Importance

                 # My loop logic:
                 # 1. "MODEL_PATH =" is in Cell 0. So I add Markdown "## 1. Config..." then Cell 0.
                 # 2. Cell 1 (Load Model) doesn't have specific keyword I checked well.
                 pass

    # Let's Rewrite the logic to be index based, it's safer for this specific file structure
    final_cells = [header_cell]

    # Original Cell 0: Imports & Config
    final_cells.append(nb['cells'][0])

    # Before Cell 1: Load
    final_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Carga del Modelo y Datos de Prueba"]
    })
    final_cells.append(nb['cells'][1])

    # Before Cell 2: Predict
    final_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Generación de Predicciones\n",
            "Aplicamos el modelo sobre el set de evaluación para obtener etiquetas y probabilidades."
        ]
    })
    final_cells.append(nb['cells'][2])

    # Before Cell 3: Confusion Matrix
    final_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Análisis de Errores (Matriz de Confusión)\n",
            "Visualizamos la distribución de aciertos y fallos. Nos interesa minimizar los Falsos Negativos (pacientes enfermos diagnosticados como sanos)."
        ]
    })
    final_cells.append(nb['cells'][3])

    # Before Cell 4: Metrics
    final_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Métricas de Desempeño\n",
            "- **Recall**: Crítico para tamizaje médico.\n",
            "- **AUC**: Medida de separabilidad entre clases."
        ]
    })
    final_cells.append(nb['cells'][4])

    # Before Cell 5: Feature Importance
    final_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Explicabilidad del Modelo\n",
            "Identificamos los factores de riesgo más importantes según el modelo aprendido."
        ]
    })
    final_cells.append(nb['cells'][5])

    nb['cells'] = final_cells

    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    update_notebook("notebooks/03_Model_Evaluation.ipynb")
