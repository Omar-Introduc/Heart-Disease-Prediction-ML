# Guía Paso a Paso del Proyecto (Step-by-Step Guide)

**Objetivo:** Guiar al usuario a través de la ejecución lógica del proyecto, desde la exploración de datos hasta la inferencia con el modelo final.

---

## Paso 1: Exploración de Datos (EDA)
**Notebook:** `notebooks/01_EDA_Clinical.ipynb`

*   **Propósito:** Entender la naturaleza de los datos clínicos de NHANES.
*   **Acciones:**
    1.  Carga de `data/02_intermediate/process_data.parquet`.
    2.  Análisis de distribuciones (histogramas de Edad, Presión, Colesterol).
    3.  Detección de valores atípicos (Outliers).
    4.  Visualización de correlaciones entre biomarcadores y la variable objetivo (`HeartDisease`).
*   **Resultado:** Comprensión de las variables críticas y validación de la calidad de los datos.

---

## Paso 2: Entrenamiento y Selección de Modelos
**Notebook:** `notebooks/02_Training_PyCaret.ipynb`

*   **Propósito:** Entrenar múltiples modelos, compararlos y optimizar el mejor.
*   **Acciones:**
    1.  **Setup de PyCaret:** Configuración del entorno, imputación automática y normalización (`normalize=True`).
    2.  **Compare Models:** Evaluación de algoritmos (LightGBM, XGBoost, RF, etc.). Se prioriza la métrica **Recall**.
    3.  **Tuning:** Optimización profunda (`n_iter=50`) de los hiperparámetros del modelo ganador (XGBoost).
    4.  **Finalize:** Entrenamiento del modelo final con todo el dataset.
    5.  **Save:** Guardado del pipeline en `models/final_pipeline_v1.pkl`.
*   **Resultado:** Un modelo serializado optimizado para alta sensibilidad.

---

## Paso 3: Evaluación del Modelo
**Notebook:** `notebooks/03_Model_Evaluation.ipynb`

*   **Propósito:** Validar el rendimiento del modelo con métricas detalladas y gráficos.
*   **Acciones:**
    1.  Carga del modelo entrenado.
    2.  Generación de la **Matriz de Confusión** (análisis de Falsos Negativos).
    3.  Curva ROC y Precision-Recall.
    4.  Análisis de Importancia de Variables (Feature Importance).
*   **Resultado:** Confirmación de que el modelo cumple con los requisitos clínicos (Recall > 0.95).

---

## Paso 4: Inferencia y Demo
**Notebook:** `notebooks/04_Inference_Demo.ipynb`

*   **Propósito:** Simular el uso del modelo en un entorno real.
*   **Acciones:**
    1.  Creación de pacientes ficticios (diccionarios Python) con diferentes perfiles de riesgo (Sano vs Crítico).
    2.  Predicción utilizando el pipeline cargado.
    3.  Visualización de la probabilidad de riesgo.
*   **Resultado:** Demostración funcional de la capacidad predictiva.

---

## Paso 5: Interfaz de Usuario (Streamlit)
**Script:** `src/app.py`

*   **Propósito:** Interfaz gráfica para uso médico.
*   **Ejecución:**
    ```bash
    streamlit run src/app.py
    ```
*   **Funcionalidad:**
    *   Formulario de entrada para datos del paciente (Edad, Signos Vitales, Laboratorio).
    *   Botón "Calcular Riesgo".
    *   Visualización del resultado y (opcionalmente) explicación SHAP.

---

## Notas Importantes
*   **Datos:** El proyecto utiliza datos procesados de NHANES 2011-2020.
*   **Modelo:** El modelo principal es un XGBoost optimizado (`XGBClassifier`).
*   **Métrica Clave:** **Recall (Sensibilidad)**. El sistema está diseñado para no perder casos positivos, asumiendo un costo mayor por falsos negativos.
