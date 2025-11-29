# Informe Final del Proyecto: Predicción de Enfermedades Cardíacas con XGBoost

## 1. Introducción
Este proyecto tuvo como objetivo desarrollar un sistema de aprendizaje automático robusto y explicable para predecir el riesgo de enfermedades cardíacas utilizando datos clínicos (NHANES). Se implementó un enfoque dual: un prototipo académico "desde cero" para comprender los fundamentos matemáticos del XGBoost, y un pipeline productivo utilizando PyCaret y Streamlit para el despliegue final.

## 2. Estado del Arte
Se revisaron algoritmos de ensamble (Random Forest, Gradient Boosting) y literatura sobre factores de riesgo cardiovascular. XGBoost se seleccionó por su eficiencia, manejo de valores faltantes y regularización integrada, superior a implementaciones tradicionales de GBDT.

## 3. Metodología

### 3.1 Datos (NHANES)
Se migraron los datos de BRFSS a NHANES (2011-2020) para priorizar biomarcadores clínicos precisos (e.g., HbA1c, Colesterol, Enzimas Hepáticas) sobre respuestas de encuestas subjetivas.
*   **Procesamiento:** Consolidación de 4 ciclos, limpieza de datos, y mapeo de columnas (e.g., 'Enzima_ALT' -> 'ALT').
*   **Variable Objetivo:** `HeartDisease`.

### 3.2 Implementación "Desde Cero" (Scratch)
Se desarrolló una implementación educativa de XGBoost (`src/tree/xgboost_scratch.py`) que incluye:
*   Cálculo de Gradientes y Hessianos para `LogLoss`.
*   Árboles de decisión que optimizan la Ganancia Estructural basada en gradientes.
*   Parámetros de regularización (Lambda, Gamma).
Esta implementación validó la comprensión teórica del equipo.

### 3.3 Implementación Productiva (PyCaret)
Se utilizó `PyCaret` para orquestar el flujo de trabajo de ML:
*   Comparación de modelos (XGBoost, LightGBM, CatBoost).
*   Optimización de hiperparámetros enfocada en **Recall** (Sensibilidad) para minimizar Falsos Negativos.
*   Pipeline final serializado en `models/best_pipeline.pkl`.

## 4. Resultados
*   **Modelo Final:** XGBoost (optimizado).
*   **Métricas Clave:** Se priorizó el Recall. (Insertar métricas finales obtenidas en notebooks anteriores).
*   **Despliegue:** Aplicación Web en Streamlit con explicabilidad SHAP integrada, permitiendo a los usuarios entender qué factores contribuyen a su riesgo.

## 5. Análisis de Ética y Sesgos
Se realizó una auditoría de equidad (`notebooks/06_Ethics_Analysis.ipynb`) evaluando la Tasa de Falsos Negativos (FNR) por sexo y edad.
*   **Importancia:** Un FNR alto en un subgrupo significa que el modelo falla en diagnosticar enfermos en ese grupo.
*   **Resultados Preliminares:** (Basado en la ejecución del notebook) Es crucial monitorear que el modelo no tenga un desempeño degradado en mujeres o grupos minoritarios debido al desbalance histórico en datos médicos.

## 6. Conclusiones
El proyecto logró entregar un MVP funcional que equilibra el rigor académico (implementación scratch) con la utilidad práctica (app desplegable). La inclusión de análisis de ética y explicabilidad (SHAP) asegura que la herramienta sea no solo precisa, sino confiable y transparente para el uso clínico asistido.

## 7. Referencias
*   Documentación de XGBoost.
*   Datos NHANES (CDC).
*   Guías de práctica clínica cardiovascular.
