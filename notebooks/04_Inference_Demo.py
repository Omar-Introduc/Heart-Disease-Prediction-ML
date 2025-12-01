#!/usr/bin/env python
# coding: utf-8

# # 🧪 Demostración de Inferencia (Simulación de Pacientes)
#
# ## 🎯 Objetivo
# Este notebook simula el uso del modelo en un entorno de producción.
# Creamos perfiles de pacientes sintéticos con diferentes niveles de riesgo para verificar que el modelo responde de manera lógica y clínicamente coherente.
#
# ## 📝 Escenarios de Prueba
# 1. **Paciente Sano**: Valores normales en todos los biomarcadores.
# 2. **Riesgo Metabólico**: Pre-hipertensión, colesterol elevado, sobrepeso.
# 3. **Paciente Crítico**: Hipertensión severa, diabetes no controlada, obesidad mórbida.
#
# ## ⚙️ Flujo
# 1. Cargar el pipeline serializado (`.pkl`).
# 2. Definir los datos de entrada (diccionarios Python).
# 3. Ejecutar `predict_model`.
# 4. Interpretar la probabilidad de riesgo.

# ## 1. Configuración e Importaciones

# ### 🔹 Paso 1: Configuración para Inferencia
# Importamos las funciones mínimas necesarias para ejecutar el modelo en producción.
# En este script, no entrenamos nada; solo cargamos y predecimos. Esto simula lo que haría una API o una aplicación web.

# In[1]:


import pandas as pd
from pycaret.classification import load_model, predict_model

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "../models/best_pipeline"


# ## 2. Carga del Modelo Serializado

# ### 🔹 Paso 2: Carga del Pipeline de Producción
# Cargamos el archivo `best_pipeline` desde el disco.
# Este objeto contiene todo lo necesario: el escalador (RobustScaler), el imputador (si lo hubiera) y el modelo clasificador final. Es una "caja negra" lista para recibir datos crudos.

# In[2]:


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================
pipeline = load_model(MODEL_PATH)
print("Model loaded successfully.")


# ## 3. Definición de Pacientes Simulados
# Creamos perfiles clínicos específicos para testear la sensibilidad del modelo.

# ### 🔹 Paso 3: Simulación de Casos Clínicos
# Definimos manualmente un diccionario con datos de pacientes hipotéticos para probar la sensibilidad del modelo:
# 1.  **Caso Sano**: Valores dentro de rangos normales.
# 2.  **Caso Riesgo**: Valores limítrofes (ej. presión alta, colesterol límite).
# 3.  **Caso Crítico**: Valores claramente patológicos.
# Esto nos permite hacer un "Sanity Check" cualitativo del modelo.

# In[3]:


# ==========================================
# 2. DEFINE PATIENT SIMULATION (NUMERIC INPUTS)
# ==========================================
# Simulating 3 clinical profiles:
# 1. Healthy
# 2. Metabolic Risk
# 3. Critical

patients_data = [
    {
        # Paciente Sano
        'Age': 35,
        'SystolicBP': 115, 'TotalCholesterol': 160, 'LDL': 90, 'Triglycerides': 110,
        'HbA1c': 5.2, 'Glucose': 88, 'UricAcid': 5.0, 'Creatinine': 0.8,
        'BMI': 23.5, 'WaistCircumference': 82, 'Height': 175,
        'Sex': 0, 'Smoking': 0, 'Alcohol': 0, 'PhysicalActivity': 1, 'HealthInsurance': 1,
        # Missing features added with default/healthy values
        'Race': 3, 'Education': 5, 'IncomeRatio': 3.5,
        'ALT_Enzyme': 20, 'AST_Enzyme': 22, 'GGT_Enzyme': 18,
        'Albumin': 4.5, 'Potassium': 4.0, 'Sodium': 140
    },
    {
        # Paciente Riesgo Metabólico
        'Age': 55,
        'SystolicBP': 145, 'TotalCholesterol': 220, 'LDL': 150, 'Triglycerides': 180,
        'HbA1c': 6.1, 'Glucose': 115, 'UricAcid': 6.8, 'Creatinine': 1.1,
        'BMI': 32.0, 'WaistCircumference': 108, 'Height': 170,
        'Sex': 0, 'Smoking': 1, 'Alcohol': 1, 'PhysicalActivity': 0, 'HealthInsurance': 1,
        # Missing features added
        'Race': 3, 'Education': 3, 'IncomeRatio': 2.0,
        'ALT_Enzyme': 35, 'AST_Enzyme': 30, 'GGT_Enzyme': 40,
        'Albumin': 4.2, 'Potassium': 4.2, 'Sodium': 142
    },
    {
        # Paciente Crítico
        'Age': 70,
        'SystolicBP': 185, 'TotalCholesterol': 290, 'LDL': 200, 'Triglycerides': 300,
        'HbA1c': 9.0, 'Glucose': 220, 'UricAcid': 8.5, 'Creatinine': 1.8,
        'BMI': 42.0, 'WaistCircumference': 125, 'Height': 165,
        'Sex': 1, 'Smoking': 1, 'Alcohol': 1, 'PhysicalActivity': 0, 'HealthInsurance': 0,
        # Missing features added
        'Race': 4, 'Education': 2, 'IncomeRatio': 0.8,
        'ALT_Enzyme': 55, 'AST_Enzyme': 60, 'GGT_Enzyme': 85,
        'Albumin': 3.5, 'Potassium': 5.1, 'Sodium': 135
    }
]

df_patients = pd.DataFrame(patients_data)
print("Patient Profiles:")
display(df_patients)


# ## 4. Predicción e Interpretación
# Ejecutamos el modelo y formateamos la salida para fácil lectura.

# ### 🔹 Paso 4: Ejecución de Inferencia y Resultados
# Pasamos los datos simulados al modelo.
# Procesamos la salida para mostrarla de forma amigable, renombrando `Label` a `Predicted_Class` y `Score` a `Probability`.
# Verificamos si el modelo clasifica correctamente al paciente 'Crítico' como clase 1 con alta probabilidad y al 'Sano' como clase 0.

# In[4]:


# ==========================================
# 3. RUN INFERENCE
# ==========================================
print("Running prediction...")
predictions = predict_model(pipeline, data=df_patients)

# Select and rename relevant output columns for display
# PyCaret 3.x uses 'prediction_label' and 'prediction_score'
results = predictions[['prediction_label', 'prediction_score']]
results.columns = ['Predicted_Class', 'Probability']

# Add context
patient_types = ['Healthy', 'Metabolic Risk', 'Critical']
results.insert(0, 'Patient_Profile', patient_types)

display(results)
