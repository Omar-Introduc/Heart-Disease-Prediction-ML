# Diseño Técnico y Arquitectura del Sistema

## 1. Visión General
El sistema ha evolucionado hacia una arquitectura clínica robusta. Mantenemos la separación entre el componente académico (aprendizaje de algoritmos) y el componente productivo, pero el flujo de datos ahora maneja **biomarcadores continuos** provenientes de NHANES, lo que exige un preprocesamiento estadístico más riguroso.

## 2. Diagrama de Arquitectura
A continuación se presenta el flujo de datos actualizado:

```mermaid
graph TD
    subgraph Data Source
        Raw[Datos Clínicos NHANES (.XPT/.CSV)]
    end

    subgraph Academic Flow [Flujo Académico - Aprendizaje]
        CleanBasic[Limpieza & Selección de Biomarcadores]
        ScalingA[Escalamiento (StandardScaler)]
        ScratchModel[Modelo XGBoostScratch]
        Metrics[Métricas de Validación]
        Report[Informe Técnico]

        Raw --> CleanBasic
        CleanBasic --> ScalingA
        ScalingA --> ScratchModel
        ScratchModel --> Metrics
        Metrics --> Report
    end

    subgraph Productive Flow [Flujo Productivo - App]
        Imputation[Imputación Clínica (KNN/Iterative)]
        ScalingP[Escalamiento Robusto]
        FeatureEng[Ingeniería (Ratios: Colesterol/HDL)]
        PyCaret[Modelo Optimizado (PyCaret)]
        Streamlit[Interfaz Médica (Streamlit)]
        User[Profesional de Salud]

        Raw --> Imputation
        Imputation --> FeatureEng
        FeatureEng --> ScalingP
        ScalingP --> PyCaret
        PyCaret --> Streamlit
        Streamlit --> User
    end

    style Academic Flow fill:#f9f,stroke:#333,stroke-width:2px
    style Productive Flow fill:#ccf,stroke:#333,stroke-width:2px
```

## 3. Descripción de Componentes

### 3.1 Flujo Académico
Centrado en entender cómo XGBoost maneja variables continuas y relaciones no lineales.
*   **Limpieza Básica:** Filtrado de outliers fisiológicamente imposibles (ej. Presión < 0).
*   **Escalamiento:** Paso crítico añadido. Convertir biomarcadores (Glucosa, Colesterol) a una escala común (Z-score) para estabilidad numérica de gradientes.
*   **XGBoostScratch:** Implementación "Vanilla".
    *   *Foco:* Observar cómo el árbol decide los "cortes" (splits) en variables continuas como HbA1c o Presión Sistólica.

### 3.2 Flujo Productivo
Orientado a la precisión clínica y despliegue.
*   **Imputación:** Manejo de exámenes de laboratorio faltantes usando correlaciones entre biomarcadores.
*   **Ingeniería de Características:** Creación de ratios médicos (ej. Índice aterogénico).
*   **PyCaret:** Entrenamiento de modelos State-of-the-Art.
*   **Streamlit:** Interfaz diseñada para ingresar valores exactos de laboratorio.

## 4. Contratos de Software
Se mantiene la interfaz `HeartDiseaseModel` ([ver interfaces](../src/interfaces.py)).

### Cambio Importante en Inputs
La interfaz de predicción (`predict`) ahora espera vectores numéricos flotantes normalizados, no categorías codificadas.
*   *Antes:* `[1, 0, 1]` (Edad 18-24, Hombre, Salud Buena)
*   *Ahora:* `[0.45, 1.2, -0.5]` (Edad estandarizada, Presión estandarizada, Colesterol estandarizado)

El componente `Streamlit` debe encargarse de capturar el dato "crudo" (ej. 120 mmHg) y pasarlo por el `Scaler` antes de enviarlo al modelo.
