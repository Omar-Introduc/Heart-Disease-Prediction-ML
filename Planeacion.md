gantt
    title Plan de Implementación Detallado del Proyecto
    dateFormat  YYYY-MM-DD
    axisFormat %Y-%m-%d

    %% -------------------------------------------------------------------------------------------
    %% --- Fase 1: Planificación y Fundamentos (Semanas 3-4) ---
    %% -------------------------------------------------------------------------------------------
    section Fase 1: Planificación y Fundamentos
    %% Semana 3: Kick-off y Propuesta
    Reunión Kick-off Definir roles y alcance [Todos]      :done, crit, milestone, 2025-09-01, 1d
    Configurar Git y Entorno (Conda/Poetry) [Líder Téc]   :crit, setup, 2025-09-09, 3d
    Investigar Estado del Arte (IEEE Scholar) [Ing. ML]    :invest_ml, after setup, 4d
    Investigar Contexto y Ética (Papers) [Analista IA]    :invest_ethics, after setup, 4d
    Hito: Propuesta Inicial Entregada [Coordinador]       :milestone, 2025-09-12, 0d
    %% Semana 4: Consolidación y Revisión
    Seleccionar Datasets (UCI Kaggle) [Ing. Datos]     :data_rev, after invest_ml, 5d
    Consolidar Propuesta (Markdown/Overleaf) [Documentador] :doc_prop, after data_rev, 5d
    Revisión Final de Propuesta en Equipo [Todos]         :rev_prop, after doc_prop, 1d 

    %% -------------------------------------------------------------------------------------------
    %% --- Fase 2: Preparación de Datos (Semanas 5-6) ---
    %% -------------------------------------------------------------------------------------------
    section Fase 2: Preparación de Datos
    %% Semana 5: Análisis Exploratorio
    Carga y Limpieza Inicial (Polars) [Ing. Datos]        :crit, data_load, after rev_prop, 3d
    Generar Reporte EDA (Sweetviz) [Ing. Datos]         :eda_sweetviz, after data_load, 2d
    Análisis Estadístico Profundo (Jupyter Seaborn) [Ing. ML] :eda_stats, after eda_sweetviz, 3d
    Definir Plan de Preprocesamiento [Todos]            :eda_plan, after eda_stats, 2d
    %% Semana 6: Implementación del Pipeline
    Desarrollar Script de Preprocesamiento (Polars) [Ing. Datos] :crit, preproc_script, after eda_plan, 4d
    Aplicar Selección de Features (Scikit-learn: RFE) [Ing. ML] :feature_sel, after preproc_script, 3d
    Crear Datasets Finales (Train/Test) [Ing. Datos]    :final_data, after feature_sel, 2d
    Hito: Dataset Limpio y Pipeline Funcional [Ing. Datos] :milestone, after final_data, 0d

    %% -------------------------------------------------------------------------------------------
    %% --- Fase 3: Diseño de Arquitectura (Semanas 7-8) ---
    %% -------------------------------------------------------------------------------------------
    section Fase 3: Diseño y Arquitectura
    %% Semana 7: Elaboración del Diseño Técnico
    Crear Diagrama de Arquitectura (Mermaid) [Líder Téc.] :crit, p3_arch, 2025-10-20, 3d
    Justificación Teórica de Algoritmos (XGBoost) [Ing. ML] :p3_theory, 2025-10-20, 4d
    Redactar Plan de Implementación [Coordinador]     :p3_plan, 2025-10-20, 4d
    Consolidar Documento Avance 1 [Documentador]      :crit, p3_doc, after p3_arch, 2d
    Hito: Entrega Avance 1 (Diseño Técnico) [Todos]     :milestone, 2025-10-24, 0d  
    %% Semana 8: Planificación de la Implementación
    Definir Contratos de Funciones/Módulos [Ing. ML]   :p3_contracts, after p3_doc, 3d
    Crear "Issues" en GitHub para Fase 4 [Líder Téc.]   :p3_issues, after p3_contracts, 2d
    Revisión de Pares del Plan Técnico [Todos]          :p3_review, after p3_issues, 2d

    %% -------------------------------------------------------------------------------------------
    %% --- Fase 4: Implementación del Prototipo (Semanas 9-10) ---
    %% -------------------------------------------------------------------------------------------
    section Fase 4: Implementación del Prototipo
    %% Semana 9: Construcción del Pipeline y Baseline
    Codificar Pipeline de Datos (en .py) [Ing. Datos]     :crit, p4_pipe, 2025-11-03, 5d
    Configurar Entorno (PyCaret setup) [Ing. ML]      :p4_setup, after p4_pipe, 1d
    Ejecutar Comparación (PyCaret compare_models) [Ing. ML] :crit, p4_compare, after p4_setup, 4d
    Desarrollo UI - Layout Base (Streamlit) [Des. de UI]   :p4_ui, 2025-11-03, 5d
    %% Semana 10: Análisis y Consolidación
    Análisis de Resultados Baseline (W&B) [Analista IA]   :p4_analysis, after p4_compare, 3d
    Selección de Modelos Candidatos (Top 3) [Todos]     :p4_select, after p4_analysis, 1d
    Consolidar en Notebook Funcional (Jupyter) [Líder Téc.] :crit, p4_notebook, after p4_select, 3d
    Hito: Prototipo Funcional Entregado [Coordinador]   :milestone, 2025-11-14, 0d

    %% -------------------------------------------------------------------------------------------
    %% --- Fase 5: Evaluación y Optimización (Semanas 11-12) ---
    %% -------------------------------------------------------------------------------------------
    section Fase 5: Evaluación y Optimización
    %% Semana 11: Optimización de Hiperparámetros
    Configurar Tuning (PyCaret tune_model) [Ing. ML]  :crit, p5_setup, 2025-11-17, 2d
    Ejecutar Búsqueda (Optuna) [Ing. ML]              :crit, p5_tune, after p5_setup, 5d
    Desarrollo Backend UI (Carga de .pkl) [Des. de UI]   :p5_ui_back, 2025-11-17, 5d
    Análisis Comparativo (Curvas ROC/AUC) [Analista IA] :p5_roc, after p5_tune, 2d
    %% Semana 12: Validación y Análisis Final
    Entrenamiento del Modelo Final (con mejores params) [Ing. ML] :crit, p5_final_train, after p5_roc, 3d
    Serializar Modelo y Pipeline (.pkl) [Ing. ML]           :p5_pkl, after p5_final_train, 1d
    Hito: Avance 2 Entregado (Resultados y Modelo) [Todos]    :milestone, 2025-11-28, 0d

    %% -------------------------------------------------------------------------------------------
    %% --- Fase 6: Análisis Final, UI y Redacción (Semanas 13-14) ---
    %% -------------------------------------------------------------------------------------------
    section Fase 6: Análisis Final y Redacción
    %% Semana 13: Análisis Profundo y Desarrollo UI
    Ejecutar Script SHAP y generar valores [Analista IA]     :crit, p6_shap_run, 2025-12-01, 2d
    Generar Visualizaciones Globales (Summary Plot) [Analista IA] :p6_shap_viz, after p6_shap_run, 3d
    Analizar Casos Individuales (Waterfall Plot) [Analista IA] :p6_shap_cases, after p6_shap_run, 3d
    Análisis Ético y de Sesgos [Analista IA]                  :p6_ethics, 2025-12-01, 7d
    Desarrollo Frontend de UI (Streamlit) [Des. de UI]        :crit, p6_ui, 2025-12-01, 7d
    %% Semana 14: Integración y Consolidación
    Redacción del Informe (Sección Resultados) [Documentador]:p6_report_res, after p6_shap_viz, 4d
    Integrar Gráficos SHAP en la UI [Des. de UI]              :p6_ui_integ, after p6_shap_viz, 3d
    Hito: Borrador Completo del Informe [Coordinador]         :milestone, 2025-12-12, 0d
    
    %% -------------------------------------------------------------------------------------------
    %% --- Fase 7: Entrega Final (Semana 15) ---
    %% -------------------------------------------------------------------------------------------
    section Fase 7: Entrega Final
    Limpiar y Documentar Código Fuente (Docstrings) [Líder Téc./Todos] :crit, p7_code, 2025-12-15, 3d
    Crear Diapositivas de Presentación (Canva/PPT) [Documentador/UI] :p7_slides, after p7_code, 3d
    Ensayo de Presentación en Equipo [Todos]                :p7_rehearse, after p7_slides, 2d
    Hito: Entrega y Presentación Final [Todos]              :milestone, 2025-12-19, 0d