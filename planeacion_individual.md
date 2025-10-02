gantt
    title Plan de Implementación (Enfoque en XGBoost)
    dateFormat  YYYY-MM-DD
    axisFormat %Y-%m-%d

 %% -------------------------------------------------------------------------------------------
    %% --- Fase 1: Planificación y Fundamentos (Semanas 3-4) ---
    %% -------------------------------------------------------------------------------------------
    section Fase 1: Planificación y Fundamentos
    %% Semana 3: Kick-off y Propuesta
    Reunión Kick-off Definir roles y alcance [Todos]      :done, crit, milestone, 2025-09-08, 1d
    Configurar Git y Entorno (Conda/Poetry) [Líder Téc]   :crit, setup, 2025-09-09, 3d
    Hito: Propuesta Inicial Entregada [Coordinador]       :milestone, 2025-09-12, 0d
    
    %% Semana 4: Inmersión Teórica en XGBoost
    Estudio del Paper Original de XGBoost [Ing. ML]       :crit, p1_paper, 2025-09-15, 3d
    Desarrollar Ecuaciones (Función Objetivo) [Ing. ML]   :crit, p1_math, after p1_paper, 2d
    Revisión de Papers Relacionados (Estado del Arte) [Analista IA] :p1_sota, 2025-09-15, 5d
    Presentación Interna de la Teoría de XGBoost [Ing. ML] :p1_present, after p1_math, 2d

    %% -------------------------------------------------------------------------------------------
    %% --- Fase 2: Preparación de Datos e Implementación Teórica (Semanas 5-6) ---
    %% -------------------------------------------------------------------------------------------
    section Fase 2: Preparación y Pruebas
    %% Semana 5: Análisis de Datos
    Seleccionar y Cargar Dataset (UCI, Kaggle) [Ing. Datos] :crit, p2_data, after p1_present, 2d
    Generar Reporte EDA con Sweetviz [Ing. Datos]        :p2_eda, after p2_data, 3d
    Crear Notebook de Análisis Visual (Seaborn) [Analista IA] :p2_viz, after p2_eda, 3d
    
    %% Semana 6: Implementación "Desde Cero"
    Codificar Árbol de Decisión Base (NumPy) [Ing. ML]  :crit, p2_tree, after p2_viz, 3d
    Implementar Cálculo de Gradiente/Hessiano [Ing. ML]  :crit, p2_grad, after p2_tree, 3d
    Crear Bucle de Boosting Secuencial [Ing. ML]        :crit, p2_boost, after p2_grad, 2d
    Hito: Prototipo de XGBoost 'Desde Cero' Funcional   :milestone, after p2_boost, 0d

    %% -------------------------------------------------------------------------------------------
    %% --- Fase 3: Diseño de Arquitectura (Semanas 7-8) ---
    %% -------------------------------------------------------------------------------------------
    section Fase 3: Diseño y Arquitectura
    %% Semana 7: Elaboración del Diseño Técnico (Avance 1)
    Crear Diagrama de Arquitectura (Mermaid) [Líder Téc.] :crit, p3_arch, 2025-10-12, 3d
    Redactar Sección de Fundamentos Matemáticos [Ing. ML] :crit, p3_theory, 2025-10-15, 4d
    Documentar Implementación Simplificada [Documentador] :p3_scratch_doc, after p3_theory, 2d
    Consolidar y Entregar Documento Avance 1 [Todos]      :crit, p3_doc, after p3_scratch_doc, 1d
    Hito: Entrega Avance 1 (Diseño Técnico)             :milestone, 2025-10-24, 0d
    
    %% (El resto de las fases pueden continuar como estaban, ya que se basan en la aplicación práctica)
    %% Semana 8: Planificación de la Implementación
    Definir Contratos de Funciones para PyCaret [Ing. ML] :p3_contracts, after p3_doc, 3d
    Crear "Issues" en GitHub para Fases 4-7 [Líder Téc.] :p3_issues, after p3_contracts, 2d
    Revisión de Pares del Plan Técnico y Código [Todos]   :p3_review, after p3_issues, 2d

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