# Diseño Técnico y Arquitectura del Sistema

## 1. Visión General
El sistema implementa una arquitectura dual para cumplir con los objetivos académicos y productivos del proyecto. Utiliza datos de la encuesta nacional NHANES (National Health and Nutrition Examination Survey) para predecir riesgo cardíaco mediante biomarcadores clínicos y determinantes sociales.

## 2. Diagrama de Arquitectura
El flujo de datos se divide en dos ramas principales:

```mermaid
graph TD
    subgraph Data Source [Fuente de Datos]
        Raw[Datos Crudos NHANES (.XPT)]
        ETL[Script ETL (carga.py)]
        Parquet[Datos Procesados (.parquet)]

        Raw --> ETL
        ETL --> Parquet
    end

    subgraph Academic Flow [Flujo Académico - XGBoost Scratch]
        LoadA[Carga de Datos]
        PreprocA[Preprocesamiento Manual (Numpy)]
        ScratchModel[Modelo XGBoostPropio (src/model.py)]
        Loss[Cálculo de Gradientes/Hessianos]
        MetricsA[Validación Teórica]

        Parquet --> LoadA
        LoadA --> PreprocA
        PreprocA --> ScratchModel
        ScratchModel --> Loss
        Loss --> ScratchModel
        ScratchModel --> MetricsA
    end

    subgraph Productive Flow [Flujo Productivo - PyCaret]
        Setup[PyCaret Setup (Auto-Preprocesamiento)]
        Impute[Imputación (Simple/Iterative)]
        Scale[Normalización (Z-Score)]
        Select[Selección de Modelos (Compare)]
        Tune[Optimización (Tune Model)]
        Final[Modelo Final (XGBClassifier)]
        API[Interfaz de Predicción (Streamlit)]

        Parquet --> Setup
        Setup --> Impute
        Impute --> Scale
        Scale --> Select
        Select --> Tune
        Tune --> Final
        Final --> API
    end

    style Academic Flow fill:#f9f,stroke:#333,stroke-width:2px
    style Productive Flow fill:#ccf,stroke:#333,stroke-width:2px
```

## 3. Descripción de Componentes

### 3.1 Fuente de Datos (NHANES)
*   **Origen:** Archivos SAS (`.XPT`) de los ciclos 2011-2020.
*   **Variables:** Clínicas (Presión, Colesterol, Glucosa), Demográficas (Edad, Raza, Educación) y Estilo de Vida (Fumar, Seguro).
*   **Salida:** Archivo unificado `data/02_intermediate/process_data.parquet`.

### 3.2 Flujo Académico (Comprensión)
Diseñado para demostrar el dominio de la teoría de Gradient Boosting.
*   **Implementación:** `src/model.py` y `src/tree/`.
*   **Lógica:** Implementa `LogLoss` personalizado, cálculo explícito de Gradientes ($p-y$) y Hessianos ($p(1-p)$), y construcción de árboles mediante algoritmo Greedy Exacto.
*   **Objetivo:** Validar que la matemática teórica coincide con la ejecución práctica.

### 3.3 Flujo Productivo (Despliegue)
Diseñado para el rendimiento máximo y uso clínico.
*   **Herramienta:** PyCaret.
*   **Pipeline:**
    1.  **Imputación:** Manejo automático de valores nulos (comunes en datos médicos).
    2.  **Normalización:** Estandarización de variables continuas para ayudar a la convergencia.
    3.  **Entrenamiento:** XGBClassifier optimizado para **Recall**.
*   **Artefacto:** `models/final_pipeline_v1.pkl`.

## 4. Contratos de Datos e Interfaces

### 4.1 Entradas del Modelo (Feature Space)
El modelo final espera las siguientes variables (definidas en `models/model_config.json`):

**Variables Numéricas:**
*   `Age` (Edad)
*   `SystolicBP` (Presión Sistólica)
*   `TotalCholesterol`, `LDL`, `Triglycerides` (Perfil Lipídico)
*   `HbA1c`, `Glucose` (Metabolismo)
*   `BMI`, `WaistCircumference` (Antropometría)
*   `Potassium`, `Sodium`, `Albumin` (Electrolitos/Hígado)
*   `ALT_Enzyme`, `AST_Enzyme`, `GGT_Enzyme` (Función Hepática)

**Variables Categóricas (Codificadas):**
*   `Sex`, `Race`, `Education`
*   `Smoking`, `Alcohol`, `PhysicalActivity`, `HealthInsurance`

### 4.2 Interfaz de Predicción
Se utiliza el protocolo `HeartDiseaseModel` definido en `src/interfaces.py` para asegurar que tanto el modelo "Scratch" como el de PyCaret puedan ser consumidos por la API de la misma manera:

```python
class HeartDiseaseModel(Protocol):
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...
```
