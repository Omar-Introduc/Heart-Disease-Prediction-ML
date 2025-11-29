
import json

def update_notebook(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)

    # 1. Update Header (Cell 0 - Insert at top)
    header_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🤖 Entrenamiento del Modelo Predictivo (PyCaret)\n",
            "\n",
            "## 🎯 Objetivo\n",
            "Este notebook orquesta el pipeline de entrenamiento de Machine Learning utilizando **PyCaret**.\n",
            "El objetivo es encontrar y optimizar el mejor algoritmo capaz de predecir la probabilidad de **Enfermedad Cardíaca** basándose en biomarcadores clínicos.\n",
            "\n",
            "## ⚙️ Estrategia de Modelado\n",
            "1. **Preprocesamiento Robusto**: Normalización y manejo de outliers.\n",
            "2. **Balanceo de Clases**: Uso de técnicas (SMOTE) para mitigar el desbalance entre pacientes sanos y enfermos.\n",
            "3. **Optimización de Recall**: Priorizamos la **Sensibilidad (Recall)** sobre la Precisión.\n",
            "   - *Contexto Médico*: Es peor no detectar a un enfermo (Falso Negativo) que alarmar a un sano (Falso Positivo).\n",
            "4. **Selección de Modelos**: Comparación automática de +15 algoritmos.\n",
            "\n",
            "## 📂 Entradas y Salidas\n",
            "- **Input**: `data/02_intermediate/process_data.parquet` (Datos limpios).\n",
            "- **Output**: `models/best_pipeline.pkl` (Modelo serializado listo para producción)."
        ]
    }

    # We will rebuild the cells list to ensure order
    new_cells = [header_cell]

    # Iterate through existing cells and add/replace markdown
    for cell in nb['cells']:
        source_text = "".join(cell['source'])

        # Check Configuration code block
        if "SAMPLE_FRAC =" in source_text and "DATA_PATH =" in source_text:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Configuración del Entorno\n",
                    "\n",
                    "Definimos parámetros globales.\n",
                    "- **SAMPLE_FRAC**: Porcentaje de datos a usar. Para pruebas rápidas usamos `0.5`, para el modelo final debe ser `1.0`.\n",
                    "- **Rutas**: Ubicación de datos y donde se guardarán los artefactos."
                ]
            })
            new_cells.append(cell)

        # Check Load Data code block
        elif "# 1. LOAD DATA" in source_text:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Carga y Filtrado de Datos\n",
                    "\n",
                    "Cargamos el dataset y aplicamos el esquema definido en `model_config.json`.\n",
                    "Es vital entrenar **solo** con las columnas que estarán disponibles en la aplicación final (Features + Target), descartando metadatos o IDs que causarían *data leakage*."
                ]
            })
            new_cells.append(cell)

        # Check Setup PyCaret code block
        elif "# 2. SETUP PYCARET" in source_text:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Configuración del Experimento (Setup)\n",
                    "\n",
                    "La función `setup()` inicializa el entorno de PyCaret y crea el pipeline de transformación.\n",
                    "- **normalize=True**: Escala las variables para que tengan rangos comparables. Usamos `RobustScaler` para ser resilientes a outliers.\n",
                    "- **remove_outliers=True**: Elimina anomalías estadísticas que podrían sesgar el modelo.\n",
                    "- **fix_imbalance=True**: Aplica SMOTE para generar muestras sintéticas de la clase minoritaria (Enfermos), mejorando el aprendizaje."
                ]
            })
            new_cells.append(cell)

        # Check Compare Models code block
        elif "# 3. COMPARE & TRAIN" in source_text:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Comparación y Selección de Modelos\n",
                    "\n",
                    "Entrenamos múltiples algoritmos (Logistic Regression, XGBoost, Random Forest, etc.) con validación cruzada (Cross-Validation).\n",
                    "**Métrica Clave: Recall**. Buscamos maximizar la capacidad del modelo para detectar casos positivos reales."
                ]
            })
            new_cells.append(cell)

        # Check Finalize code block
        elif "# 4. FINALIZE & SAVE" in source_text:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Finalización y Persistencia\n",
                    "\n",
                    "Una vez seleccionado el mejor modelo:\n",
                    "1. **Finalize**: Se re-entrena el modelo utilizando el 100% de los datos (incluyendo el set de prueba reservado anteriormente).\n",
                    "2. **Save**: Se guarda el pipeline completo (preprocesamiento + modelo) en un archivo `.pkl` para su despliegue en la API/Streamlit."
                ]
            })
            new_cells.append(cell)

        else:
            # Append other cells if any (though looking at the file, we covered all code blocks)
            # To be safe, avoid duplicates if I manually inserted headers inside code cells in previous attempts (which I didn't)
            # But the original file has code cells with comments. We keep them.
            pass

    nb['cells'] = new_cells

    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    update_notebook("notebooks/02_Training_PyCaret.ipynb")
