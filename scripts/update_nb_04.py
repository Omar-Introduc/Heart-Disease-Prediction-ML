import json


def update_notebook(filepath):
    with open(filepath, "r") as f:
        nb = json.load(f)

    # 1. Update Header (Cell 0)
    header_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🧪 Demostración de Inferencia (Simulación de Pacientes)\n",
            "\n",
            "## 🎯 Objetivo\n",
            "Este notebook simula el uso del modelo en un entorno de producción.\n",
            "Creamos perfiles de pacientes sintéticos con diferentes niveles de riesgo para verificar que el modelo responde de manera lógica y clínicamente coherente.\n",
            "\n",
            "## 📝 Escenarios de Prueba\n",
            "1. **Paciente Sano**: Valores normales en todos los biomarcadores.\n",
            "2. **Riesgo Metabólico**: Pre-hipertensión, colesterol elevado, sobrepeso.\n",
            "3. **Paciente Crítico**: Hipertensión severa, diabetes no controlada, obesidad mórbida.\n",
            "\n",
            "## ⚙️ Flujo\n",
            "1. Cargar el pipeline serializado (`.pkl`).\n",
            "2. Definir los datos de entrada (diccionarios Python).\n",
            "3. Ejecutar `predict_model`.\n",
            "4. Interpretar la probabilidad de riesgo.",
        ],
    }

    new_cells = [header_cell]

    for cell in nb["cells"]:
        source_text = "".join(cell["source"])

        if "MODEL_PATH =" in source_text:
            new_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 1. Carga del Pipeline\n",
                        "Importamos el modelo entrenado que contiene todos los pasos de preprocesamiento (escalado, imputación, etc) y el clasificador final.",
                    ],
                }
            )
            new_cells.append(cell)

        elif "# 1. LOAD TRAINED MODEL" in source_text:
            # Already handled/covered by logic above or redundant.
            # Original cell 1 has loading logic.
            # Let's check original structure:
            # Cell 0: Imports + Config
            # Cell 1: Load Model (predict_model imported in cell 0)
            pass

        elif "patients_data =" in source_text:
            new_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 2. Definición de Pacientes Simulados\n",
                        "\n",
                        "Creamos un DataFrame manual con 3 casos de uso típicos.\n",
                        "**Nota**: Los valores están basados en rangos clínicos reales (ej. Presión Sistólica > 140 es hipertensión).",
                    ],
                }
            )
            new_cells.append(cell)

        elif "predict_model" in source_text:
            new_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 3. Ejecución de Inferencia\n",
                        "\n",
                        "El modelo recibe los datos crudos y devuelve:\n",
                        "- **Predicted_Class**: 0 (Sano) o 1 (Riesgo).\n",
                        "- **Probability**: La confianza del modelo en su predicción.\n",
                        "\n",
                        "Esperamos que la probabilidad de enfermedad aumente conforme el perfil clínico empeora.",
                    ],
                }
            )
            new_cells.append(cell)
        else:
            # Handle remaining cells.
            # Original Cell 0: Imports
            # Original Cell 1: Load Model code
            if "import pandas" in source_text:
                # Already added if we match MODEL_PATH above? No, imports are cell 0. MODEL_PATH is cell 0.
                # Ah, wait. In 04_Inference_Demo.ipynb:
                # Cell 0: Imports + Config (MODEL_PATH)
                # Cell 1: Load Model Code
                pass

    # Re-doing list construction logic to be robust based on cell content scanning/index
    # Cell 0 (Imports) is usually preserved.
    # We will build the new list explicitly.

    final_cells = [header_cell]

    # Cell 0: Config/Imports
    final_cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Configuración e Importaciones"],
        }
    )
    final_cells.append(nb["cells"][0])

    # Cell 1: Load Model
    final_cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Carga del Modelo Serializado"],
        }
    )
    final_cells.append(nb["cells"][1])

    # Cell 2: Patient Data
    final_cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Definición de Pacientes Simulados\n",
                "Creamos perfiles clínicos específicos para testear la sensibilidad del modelo.",
            ],
        }
    )
    final_cells.append(nb["cells"][2])

    # Cell 3: Prediction
    final_cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Predicción e Interpretación\n",
                "Ejecutamos el modelo y formateamos la salida para fácil lectura.",
            ],
        }
    )
    final_cells.append(nb["cells"][3])

    nb["cells"] = final_cells

    with open(filepath, "w") as f:
        json.dump(nb, f, indent=1)


if __name__ == "__main__":
    update_notebook("notebooks/04_Inference_Demo.ipynb")
