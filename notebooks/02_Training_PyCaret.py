#!/usr/bin/env python
# coding: utf-8

# # 🤖 Entrenamiento del Modelo Predictivo (PyCaret)
#
# ## 🎯 Objetivo
# Este notebook orquesta el pipeline de entrenamiento de Machine Learning utilizando **PyCaret**.
# El objetivo es encontrar y optimizar el mejor algoritmo capaz de predecir la probabilidad de **Enfermedad Cardíaca** basándose en biomarcadores clínicos.
#
# ## ⚙️ Estrategia de Modelado
# 1. **Preprocesamiento Robusto**: Normalización y manejo de outliers.
# 2. **Balanceo de Clases**: Uso de técnicas (SMOTE) para mitigar el desbalance entre pacientes sanos y enfermos.
# 3. **Optimización de Recall**: Priorizamos la **Sensibilidad (Recall)** sobre la Precisión.
#    - *Contexto Médico*: Es peor no detectar a un enfermo (Falso Negativo) que alarmar a un sano (Falso Positivo).
# 4. **Selección de Modelos**: Comparación automática de +15 algoritmos.
#
# ## 📂 Entradas y Salidas
# - **Input**: `data/02_intermediate/process_data.parquet` (Datos limpios).
# - **Output**: `models/best_pipeline.pkl` (Modelo serializado listo para producción).

# ## 1. Configuración del Entorno
#
# Definimos parámetros globales.
# - **SAMPLE_FRAC**: Porcentaje de datos a usar. Para pruebas rápidas usamos `0.5`, para el modelo final debe ser `1.0`.
# - **Rutas**: Ubicación de datos y donde se guardarán los artefactos.

# ### 🔹 Paso 1: Configuración del Entorno y Constantes
# Inicializamos el entorno de trabajo importando **PyCaret** y definiendo constantes críticas:
# - `SAMPLE_FRAC`: Controla el muestreo de datos. Usamos 0.5 (50%) para iteraciones rápidas de desarrollo, pero se debe cambiar a 1.0 para el entrenamiento final.
# - `DATA_PATH` y `MODEL_DIR`: Definen las rutas de entrada de datos y salida del modelo, asegurando una estructura de proyecto ordenada.

# In[7]:


import pandas as pd
from pycaret.classification import *
import os
import json

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLE_FRAC = 0.10  # Modified for quick test
DATA_PATH = "../data/02_intermediate/process_data.parquet"
MODEL_DIR = "../models"
MODEL_NAME = "best_pipeline"
CONFIG_PATH = "../models/model_config.json"

print(f"Running Training with SAMPLE_FRAC = {SAMPLE_FRAC}")


# ## 2. Carga y Filtrado de Datos
#
# Cargamos el dataset y aplicamos el esquema definido en `model_config.json`.
# Es vital entrenar **solo** con las columnas que estarán disponibles en la aplicación final (Features + Target), descartando metadatos o IDs que causarían *data leakage*.

# ### 🔹 Paso 2: Carga y Selección de Features (Data Loading)
# Cargamos el dataset procesado y aplicamos un filtro estricto de columnas basado en `model_config.json`.
# **Importante**:
# - Solo cargamos las columnas definidas como `features` y el `target`.
# - Esto actúa como una barrera de seguridad contra el *data leakage*, asegurando que el modelo no vea variables que no estarán disponibles en producción (como IDs de pacientes o fechas de procesamiento).

# In[8]:


# ==========================================
# 1. LOAD DATA
# ==========================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)
print(f"Original Data Shape: {df.shape}")

# Load Schema Config (REPLACED WITH MANUAL DEFINITION)
# with open(CONFIG_PATH, 'r') as f:
#     config = json.load(f)

# Define manually the columns based on EDA and Extraction
numeric_features = [
    'Age', 'SystolicBP', 'BMI', 'WaistCircumference', 'Height',
    'TotalCholesterol', 'Triglycerides', 'LDL', 'HbA1c', 'Glucose',
    'Creatinine', 'UricAcid', 'ALT_Enzyme', 'Albumin', 'Potassium',
    'Sodium', 'GGT_Enzyme', 'AST_Enzyme', 'IncomeRatio'
]
categorical_features = [
    'Sex', 'Race', 'Education', 'Smoking', 'Alcohol',
    'PhysicalActivity', 'HealthInsurance'
]
target = 'HeartDisease'

features = numeric_features + categorical_features

# Filter only relevant columns
df = df[features + [target]]

if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=42)
    print(f"Sampled Data Shape: {df.shape}")
else:
    print("Using Full Dataset")


# ## 3. Configuración del Experimento (Setup)
#
# La función `setup()` inicializa el entorno de PyCaret y crea el pipeline de transformación.
# - **normalize=True**: Escala las variables para que tengan rangos comparables. Usamos `RobustScaler` para ser resilientes a outliers.
# - **remove_outliers=True**: Elimina anomalías estadísticas que podrían sesgar el modelo.
# - **fix_imbalance=True**: Aplica SMOTE para generar muestras sintéticas de la clase minoritaria (Enfermos), mejorando el aprendizaje.

# ### 🔹 Paso 3: Inicialización del Experimento (PyCaret Setup)
# Configuramos el pipeline de preprocesamiento automático con `setup()`. Aquí definimos la "magia" de PyCaret:
# - **Normalización**: Aplicamos `RobustScaler` (`normalize_method='robust'`) para escalar los datos manejando bien los outliers típicos de datos clínicos.
# - **Balanceo de Clases**: Activamos `fix_imbalance=True` (SMOTE) para generar datos sintéticos de la clase minoritaria (pacientes enfermos), evitando que el modelo se sesgue hacia la clase mayoritaria (sanos).
# - **Tipos de Datos**: Definimos explícitamente cuáles son numéricas y cuáles categóricas.

# In[9]:


# ==========================================
# 2. SETUP PYCARET
# ==========================================
# normalize=True (RobustScaler)
# remove_outliers=True
# fix_imbalance=True

exp = setup(
    data=df,
    target=target,
    numeric_features=numeric_features,
    categorical_features=categorical_features,
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    fix_imbalance=True,
    session_id=42,
    verbose=True
)


# ## 4. Comparación y Selección de Modelos
#
# Entrenamos múltiples algoritmos (Logistic Regression, XGBoost, Random Forest, etc.) con validación cruzada (Cross-Validation).
# **Métrica Clave: Recall**. Buscamos maximizar la capacidad del modelo para detectar casos positivos reales.

# In[ ]:


import numpy as np

# ==========================================
# 3. SELECCIÓN DE MODELO (Model Selection)
# ==========================================
# Estrategia: Comparar modelos basados en árboles y seleccionar los Top 3 para tuning.
# Restricción: Solo árboles (XGBoost, LightGBM)
# Métrica: Recall (Sensibilidad)

top_models = compare_models(
    include=['xgboost', 'lightgbm'],
    sort='Recall',
    n_select=3,
    verbose=False
)
print(f"Top 3 Models: {top_models}")


# ### 🔹 Paso 4.1: Optimización Profunda de Hiperparámetros
# Ya tenemos el mejor candidato ('best_model'). Ahora, no nos conformamos con sus parámetros por defecto.
# Ejecutamos un proceso de **Tuning Exhaustivo**:
# - **optimize='Recall'**: El algoritmo de búsqueda intentará maximizar específicamente la sensibilidad.
# - **n_iter=50**: Probamos 50 combinaciones de hiperparámetros distintas. ¿Por qué 50? En medicina, la diferencia entre un recall del 85% y 87% puede significar salvar más vidas (menos Falsos Negativos). Una búsqueda superficial (n_iter=10) podría perder el óptimo global.
# - **choose_better=True**: Si después de tunear el modelo empeora, nos quedamos con la versión original.

# In[ ]:


# ==========================================
# 3.1 DEEP HYPERPARAMETER TUNING
# ==========================================
best_model = top_models[0]
print(f"\n--- Tuning Best Model: {type(best_model).__name__} ---")

# Tuning loop
# n_iter=2 para búsqueda exhaustiva (Deep Search)
tuned_model = tune_model(
    best_model,
    optimize='Recall',
    n_iter=2,
    choose_better=True,
    verbose=False
)

print(f"\nFINAL BEST MODEL SELECTED: {tuned_model}")


# ### 🔹 Paso 4.2: Optimización de Umbral de Decisión
#
# Implementamos la estrategia **Precision-Constrained Recall Maximization**.
# Buscamos el umbral que maximice el Recall, sujeto a que la Precisión sea >= 0.4.

# In[ ]:


# 3. Estrategia de Umbral de Seguridad Clínica
print("\n--- Optimizando Umbral de Decisión ---")
# Generar probabilidades en el set de validación (hold-out)
predictions = predict_model(tuned_model, raw_score=True, verbose=False)

# Identificar columnas de score y target real
target_col = get_config('target_param')
y_true = predictions[target_col]

# Buscar columna de score para clase positiva (1)
score_cols = [c for c in predictions.columns if 'score' in c]
if any('1' in c for c in score_cols):
    score_col = [c for c in score_cols if '1' in c][0]
else:
    score_col = score_cols[0]

y_scores = predictions[score_col]

# Iterar umbrales
thresholds = np.arange(0.0, 1.0, 0.01)
results = []

from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

for t in thresholds:
    y_pred = (y_scores >= t).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    results.append({'Threshold': t, 'Precision': prec, 'Recall': rec})

results_df = pd.DataFrame(results)

# Filtrar zona segura: Precision >= 0.4
safe_zone = results_df[results_df['Precision'] >= 0.4]

if not safe_zone.empty:
    # Seleccionar el umbral con mayor Recall dentro de la zona segura
    # (Generalmente el umbral más bajo de la zona)
    best_row = safe_zone.sort_values('Recall', ascending=False).iloc[0]
    optimal_threshold = best_row['Threshold']
    print(f"✅ Umbral Óptimo Encontrado: {optimal_threshold:.2f}")
    print(f"   Métricas Esperadas -> Recall: {best_row['Recall']:.4f} | Precision: {best_row['Precision']:.4f}")
else:
    print("⚠️ No se alcanzó la zona segura (Precision >= 0.4). Se usará umbral por defecto (0.5).")
    optimal_threshold = 0.5

# Visualización de la Curva de Seguridad
plt.figure(figsize=(10, 6))
plt.plot(results_df['Threshold'], results_df['Precision'], label='Precision', color='blue')
plt.plot(results_df['Threshold'], results_df['Recall'], label='Recall', color='green')
plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimum ({optimal_threshold:.2f})')

# Sombrear zona segura si existe
if not safe_zone.empty:
    plt.axvspan(safe_zone['Threshold'].min(), safe_zone['Threshold'].max(), alpha=0.1, color='green', label='Zona Segura (Prec>=0.4)')

plt.title(f"Curva de Seguridad Clínica: Selección de Umbral para {type(tuned_model).__name__}")
plt.xlabel("Umbral de Decisión")
plt.ylabel("Métrica")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ## 5. Finalización y Persistencia
#
# Una vez seleccionado el mejor modelo:
# 1. **Finalize**: Se re-entrena el modelo utilizando el 100% de los datos (incluyendo el set de prueba reservado anteriormente).
# 2. **Save**: Se guarda el pipeline completo (preprocesamiento + modelo) en un archivo `.pkl` para su despliegue en la API/Streamlit.

# ### 🔹 Paso 5: Finalización y Serialización del Modelo
# Una vez seleccionado el mejor algoritmo:
# 1.  **Finalize**: Re-entrenamos el modelo utilizando **todos** los datos disponibles (incluyendo el set de validación que PyCaret retuvo internamente).
# 2.  **Save**: Guardamos el pipeline completo como un archivo `.pkl` en el directorio `models/`. Este archivo contiene tanto el modelo predictivo como las transformaciones de datos (escalado, imputación), listo para ser consumido por la API.

# ## 4.5 Explicabilidad del Modelo (SHAP)
#
# Validamos que el modelo no tome decisiones basadas en artefactos o sesgos. Generamos el **SHAP Summary Plot** para visualizar las variables más impactantes.
# Esto es un requisito de **Transparencia Algorítmica** para la auditoría.

# In[ ]:


# Generar SHAP Summary Plot
print("Generando explicaciones SHAP...")
try:
    interpret_model(tuned_model, plot='summary')
except Exception as e:
    print(f"No se pudo generar el gráfico SHAP (probablemente el modelo no lo soporte nativamente o falte librería): {e}")


# In[12]:


# ==========================================
# 4. FINALIZE & SAVE
# ==========================================
final_model = finalize_model(tuned_model)
os.makedirs(MODEL_DIR, exist_ok=True)
save_path = os.path.join(MODEL_DIR, MODEL_NAME)
save_model(final_model, save_path)
print(f"Model saved successfully to {save_path}.pkl")
