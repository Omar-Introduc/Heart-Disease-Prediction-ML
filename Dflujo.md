%% Diagrama de Flujo del Proyecto de IA para Predicción de Riesgo Cardíaco
graph TD
    subgraph "Fase 1: Entrada y Preparación"
        A[/"Dataset Crudo (.csv)"/] --> B{Cargar y Limpiar Datos};
        B --> C["Análisis Exploratorio (EDA)"];
        C --> D["Dataset Preparado"];
    end

    subgraph "Fase 2: Modelado y Optimización"
        D --> E{Entrenamiento y Comparación de Modelos};
        E --> F{"¿Rendimiento Aceptable?"};
        F -- No --> G[Ajuste de Hiperparámetros];
        G --> E;
        F -- Sí --> H["Modelo Final Optimizado"];
    end

    subgraph "Fase 3: Interpretación y Despliegue"
        H --> I{Análisis de Interpretabilidad};
        I --> J[/"Dashboard Interactivo"/];
        J --> K[("Predicción de Riesgo")];
    end

    %% --- Estilos y Herramientas Específicas ---
    subgraph "Stack Tecnológico"
        direction LR
        tool1[Python];
        tool2[Polars];
        tool3[Sweetviz];
        tool4[PyCaret];
        tool5[Optuna];
        tool6[SHAP];
        tool7[Streamlit];
    end

    B -- "Usa" --> tool2;
    C -- "Usa" --> tool3;
    E -- "Usa" --> tool4;
    G -- "Usa" --> tool5;
    I -- "Usa" --> tool6;
    J -- "Construido con" --> tool7;

    style A fill:#f9f,stroke:#333,stroke-width:2px;
    style D fill:#ccf,stroke:#333,stroke-width:2px;
    style H fill:#9f9,stroke:#333,stroke-width:2px;
    style K fill:#f66,stroke:#333,stroke-width:2px;
    style F fill:#f90,stroke:#333,stroke-width:2px;


____________________________________________________________________________________________________________________________________________________

%% Arquitectura de Sistema de ML con enfoque AutoML
graph TD
    subgraph "Fase 1: Preparación y Análisis de Datos"
        A[/"Dataset Crudo (.csv)"/] --> B["Carga y Limpieza con Polars"];
        B --> C["Análisis Exploratorio con Sweetviz"];
        C --> D["Dataset Limpio y Preparado"];
    end

    subgraph "Fase 2: Experimentación y Optimización AutoML"
        D --> E["Setup del Entorno PyCaret"];
        E --> F["Comparación de Modelos (compare_models)"];
        
        subgraph "Modelos en Competencia"
            direction LR
            M1[Regresión Logística];
            M2[Random Forest];
            M3[XGBoost];
            M_etc[...]
        end

        F --> G{"Selección del Mejor Modelo (Basado en AUC/Recall)"};
        G --> H["Afinamiento de Hiperparámetros (tune_model con Optuna)"];
        H --> I["Modelo Final Optimizado (.pkl)"];
    end

subgraph "Fase 3: Interpretación y Despliegue"
        I["Modelo Final Optimizado (.pkl)"] --> J["Interpretación del Modelo con SHAP"];
        J --> K["Integrar Modelo (.pkl) en Dashboard"];
        K --> L["Interfaz de Usuario con Streamlit"];
        L --> M[("Predicción Final para el Usuario")];
    end

    %% Conexiones internas
    E --> M1 & M2 & M3 & M_etc;
    
    %% Estilos
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#ccf,stroke:#333,stroke-width:2px
    style I fill:#9f9,stroke:#333,stroke-width:2px
    style M fill:#f66,stroke:#333,stroke-width:2px
    style F fill:#f90,stroke:#333,stroke-width:2px