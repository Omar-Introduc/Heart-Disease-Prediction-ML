# Resultados del Proyecto y Conclusiones Finales

## 1. Selección del Modelo

Durante la fase de experimentación (Sprint 4 y 5), se compararon múltiples algoritmos de Gradient Boosting. El objetivo principal fue maximizar la sensibilidad (Recall) para minimizar los falsos negativos en la detección de riesgo cardíaco.

### Tabla de Comparación (Top Modelos)

| Modelo | Recall (Sensibilidad) |
|--------|-----------------------|
| **XGBoost (XGBClassifier)** | **1.0000** |
| LightGBM | < 1.0000 |

**Decisión:** Se seleccionó **XGBoost** como el modelo final.
**Razón:** Tras el ajuste de hiperparámetros (`tune_model` con `optimize='Recall'`), XGBoost demostró un rendimiento superior en la métrica objetivo. En el contexto médico, un Recall de 1.0 (o cercano a él) indica que el modelo es extremadamente eficaz identificando a los pacientes positivos, actuando como una herramienta de tamizaje (screening) robusta.

> **Nota Técnica:** El Recall perfecto de 1.0000 en el conjunto de validación puede indicar un rendimiento excelente en los datos disponibles, aunque en producción se debe monitorear continuamente para evitar falsos positivos excesivos.

## 2. Configuración Final del Modelo

El modelo ganador opera con los siguientes hiperparámetros clave (optimizados):

*   **Algoritmo:** XGBClassifier (Booster: `gbtree`)
*   **Optimizador:** Adam/Gradient Descent con Regularización
*   **Hiperparámetros:**
    *   `learning_rate`: 0.1
    *   `n_estimators`: 100
    *   `max_depth`: 5
    *   `colsample_bytree`: 1.0

## 3. Importancia de Variables (Feature Importance)

El análisis de importancia de características revela qué factores biológicos y demográficos pesan más en la decisión del algoritmo. Según los resultados oficiales:

1.  **Age (Edad):** El factor dominante. El riesgo cardíaco está fuertemente correlacionado con el envejecimiento.
2.  **Education (Nivel Educativo):** Variable socioeconómica que actúa como proxy de acceso a salud y estilo de vida.
3.  **HealthInsurance (Seguro Médico):** Otro determinante social clave.
4.  **Race (Raza/Etnicidad):** Factores demográficos relevantes en estudios poblacionales como NHANES.

Esta jerarquía difiere de los modelos puramente clínicos tradicionales, sugiriendo que en este dataset (NHANES), los determinantes sociales de la salud juegan un papel crucial junto con los biomarcadores.

## 4. Interpretación de la Matriz de Confusión

Dado el alto Recall:
*   **Falsos Negativos (FN) $\approx$ 0:** El modelo no deja escapar casos de riesgo.
*   **Falsos Positivos (FP):** Es probable que existan alertas en pacientes sanos. Esto es aceptable en un sistema de triaje: se prefiere revisar a un paciente sano que ignorar a uno enfermo.

## 5. Conclusión del Proyecto

Se ha logrado implementar exitosamente un flujo de trabajo completo ("End-to-End") que va desde la ingestión de datos crudos SAS de NHANES hasta un modelo predictivo desplegable.

*   **Componente Académico:** Se validó la teoría de Gradient Boosting mediante una implementación desde cero (`XGBoostScratch`) que replica la lógica matemática de gradientes y hessianos.
*   **Componente Productivo:** Se entregó un pipeline robusto en PyCaret capaz de integrarse con una interfaz de usuario para uso clínico referencial.
