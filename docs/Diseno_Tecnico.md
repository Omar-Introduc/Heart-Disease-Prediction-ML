# Diseño Técnico y Arquitectura del Sistema

## 1. Visión General
El sistema está diseñado con una arquitectura bifurcada que separa el componente académico (implementación desde cero para aprendizaje) del componente productivo (solución optimizada para despliegue).

## 2. Diagrama de Arquitectura
A continuación se presenta el flujo de datos y componentes del sistema:

```mermaid
graph TD
    subgraph Data Source
        Raw[Datos Raw (.XPT)]
    end

    subgraph Academic Flow [Flujo Académico - Aprendizaje]
        CleanBasic[Limpieza Básica]
        ScratchModel[Modelo XGBoostScratch]
        Metrics[Métricas de Validación]
        Report[Informe Técnico]

        Raw --> CleanBasic
        CleanBasic --> ScratchModel
        ScratchModel --> Metrics
        Metrics --> Report
    end

    subgraph Productive Flow [Flujo Productivo - App]
        FeatureEng[Ingeniería de Características]
        PyCaret[Modelo Optimizado (PyCaret)]
        Streamlit[Interfaz de Usuario (Streamlit)]
        User[Usuario Final]

        Raw --> FeatureEng
        FeatureEng --> PyCaret
        PyCaret --> Streamlit
        Streamlit --> User
    end

    style Academic Flow fill:#f9f,stroke:#333,stroke-width:2px
    style Productive Flow fill:#ccf,stroke:#333,stroke-width:2px
```

## 3. Descripción de Componentes

### 3.1 Flujo Académico
Este flujo justifica el Sprint 2 y parte del Sprint 3.
*   **Limpieza Básica:** Transformación mínima necesaria para que el algoritmo funcione (manejo de nulos, codificación numérica).
*   **XGBoostScratch:** Implementación "Vanilla" del algoritmo Gradient Boosting Tree.
    *   *Objetivo:* Demostrar comprensión profunda de la matemática (Gradientes, Hessianos, Ganancia).
    *   *Limitación:* No apto para grandes volúmenes de datos en tiempo real (Algoritmo Exact Greedy).

### 3.2 Flujo Productivo
Este flujo será el foco del Sprint 4 en adelante.
*   **Ingeniería de Características:** Pipeline robusto (imputación avanzada, selección de variables, manejo de outliers).
*   **PyCaret:** Uso de librerías industriales para selección de modelos, tuning de hiperparámetros y ensamblaje.
*   **Streamlit:** Interfaz de usuario que abstrae la complejidad del modelo.

## 4. Contratos de Software
Para asegurar que la interfaz de usuario (Streamlit) sea agnóstica al modelo subyacente (aunque usaremos principalmente el productivo), se define una interfaz común `HeartDiseaseModel`.

Ver [Interfaces](../src/interfaces.py) y documentación de implementación.
