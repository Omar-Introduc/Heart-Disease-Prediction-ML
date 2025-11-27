# Planeación del Proyecto (Formato Sprint/Issue)

Este documento estructura el plan de implementación del proyecto (originalmente en diagrama de Gantt) en Sprints de trabajo con sus respectivos Issues (Tareas).

## Sprint 1: Planificación y Fundamentos (Semanas 3-4)
**Objetivo:** Establecer las bases del proyecto, definir roles y adquirir el conocimiento teórico necesario sobre XGBoost.

*   **Issue 1: Reunión Kick-off**
    *   **Descripción:** Definir roles y alcance del proyecto.
    *   **Asignado a:** Todos
    *   **Fecha estimada:** Semana 3

*   **Issue 2: Configuración del Entorno**
    *   **Descripción:** Configurar Git y el entorno de desarrollo (Conda/Poetry).
    *   **Asignado a:** Líder Técnico
    *   **Fecha estimada:** Semana 3

*   **Issue 3: Entrega de Propuesta Inicial**
    *   **Descripción:** Finalizar y entregar la propuesta inicial del proyecto.
    *   **Asignado a:** Coordinador
    *   **Fecha estimada:** Semana 3 (Hito)

*   **Issue 4: Estudio Teórico de XGBoost**
    *   **Descripción:** Estudiar el Paper Original de XGBoost.
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 4

*   **Issue 5: Desarrollo Matemático**
    *   **Descripción:** Desarrollar y documentar las ecuaciones (Función Objetivo).
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 4

*   **Issue 6: Estado del Arte**
    *   **Descripción:** Revisión de Papers Relacionados y estado del arte.
    *   **Asignado a:** Analista IA
    *   **Fecha estimada:** Semana 4

*   **Issue 7: Presentación Teórica**
    *   **Descripción:** Presentación Interna de la Teoría de XGBoost al equipo.
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 4

---

## Sprint 2: Preparación y Pruebas (Semanas 5-6)
**Objetivo:** Preparar los datos y realizar una implementación "desde cero" de los componentes clave de XGBoost para comprender su funcionamiento.

*   **Issue 8: Ingesta de Datos**
    *   **Descripción:** Seleccionar y cargar el Dataset (UCI, Kaggle).
    *   **Asignado a:** Ing. Datos
    *   **Fecha estimada:** Semana 5

*   **Issue 9: Análisis Exploratorio Automático**
    *   **Descripción:** Generar reporte EDA con herramientas como Sweetviz.
    *   **Asignado a:** Ing. Datos
    *   **Fecha estimada:** Semana 5

*   **Issue 10: Análisis Visual Profundo**
    *   **Descripción:** Crear Notebook de análisis visual detallado (usando Seaborn/Matplotlib).
    *   **Asignado a:** Analista IA
    *   **Fecha estimada:** Semana 5

*   **Issue 11: Árbol de Decisión Base**
    *   **Descripción:** Codificar un Árbol de Decisión base utilizando solo NumPy.
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 6

*   **Issue 12: Gradientes y Hessianos**
    *   **Descripción:** Implementar el cálculo de Gradiente y Hessiano para la función de pérdida.
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 6

*   **Issue 13: Boosting Secuencial**
    *   **Descripción:** Crear el bucle de Boosting secuencial para entrenar el modelo.
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 6

*   **Issue 14: Prototipo Desde Cero**
    *   **Descripción:** Validar que el prototipo de XGBoost 'Desde Cero' sea funcional.
    *   **Asignado a:** Todos (Hito)
    *   **Fecha estimada:** Semana 6

---

## Sprint 3: Diseño y Arquitectura (Semanas 7-8)
**Objetivo:** Formalizar el diseño técnico del sistema y planificar la implementación productiva.

*   **Issue 15: Arquitectura del Sistema**
    *   **Descripción:** Crear Diagrama de Arquitectura (usando Mermaid u otra herramienta).
    *   **Asignado a:** Líder Técnico
    *   **Fecha estimada:** Semana 7

*   **Issue 16: Documentación Matemática**
    *   **Descripción:** Redactar la sección de fundamentos matemáticos para el informe.
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 7

*   **Issue 17: Documentación de Implementación**
    *   **Descripción:** Documentar la implementación simplificada realizada en el Sprint anterior.
    *   **Asignado a:** Documentador
    *   **Fecha estimada:** Semana 7

*   **Issue 18: Entrega Avance 1**
    *   **Descripción:** Consolidar y entregar el documento de Avance 1 (Diseño Técnico).
    *   **Asignado a:** Todos (Hito)
    *   **Fecha estimada:** Semana 7

*   **Issue 19: Contratos de Software**
    *   **Descripción:** Definir contratos de funciones e interfaces para la integración con PyCaret.
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 8

*   **Issue 20: Gestión de Tareas**
    *   **Descripción:** Crear "Issues" detallados en GitHub para las fases restantes (4-7).
    *   **Asignado a:** Líder Técnico
    *   **Fecha estimada:** Semana 8

*   **Issue 21: Revisión Técnica**
    *   **Descripción:** Revisión de pares del plan técnico y del código existente.
    *   **Asignado a:** Todos
    *   **Fecha estimada:** Semana 8

---

## Sprint 4: Implementación del Prototipo (Semanas 9-10)
**Objetivo:** Construir el pipeline de datos, establecer un baseline con PyCaret y crear la primera versión de la UI.

*   **Issue 22: Pipeline de Datos**
    *   **Descripción:** Codificar el pipeline de procesamiento de datos (scripts .py).
    *   **Asignado a:** Ing. Datos
    *   **Fecha estimada:** Semana 9

*   **Issue 23: Setup de PyCaret**
    *   **Descripción:** Configurar el entorno y experimentos en PyCaret.
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 9

*   **Issue 24: Comparación de Modelos**
    *   **Descripción:** Ejecutar `compare_models` en PyCaret para establecer baselines.
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 9

*   **Issue 25: Layout UI Base**
    *   **Descripción:** Desarrollo del layout base de la interfaz de usuario con Streamlit.
    *   **Asignado a:** Des. de UI
    *   **Fecha estimada:** Semana 9

*   **Issue 26: Análisis de Resultados Baseline**
    *   **Descripción:** Analizar resultados de los modelos baseline (usando W&B si aplica).
    *   **Asignado a:** Analista IA
    *   **Fecha estimada:** Semana 10

*   **Issue 27: Selección de Modelos**
    *   **Descripción:** Seleccionar los modelos candidatos (Top 3) para optimización.
    *   **Asignado a:** Todos
    *   **Fecha estimada:** Semana 10

*   **Issue 28: Notebook Funcional**
    *   **Descripción:** Consolidar el trabajo en un Notebook Funcional (Jupyter).
    *   **Asignado a:** Líder Técnico
    *   **Fecha estimada:** Semana 10

*   **Issue 29: Entrega Prototipo Funcional**
    *   **Descripción:** Hito de entrega del prototipo funcional.
    *   **Asignado a:** Coordinador (Hito)
    *   **Fecha estimada:** Semana 10

---

## Sprint 5: Evaluación y Optimización (Semanas 11-12)
**Objetivo:** Optimizar los modelos seleccionados, integrar el backend de la UI y validar los resultados finales.

*   **Issue 30: Configuración de Tuning**
    *   **Descripción:** Configurar el proceso de tuning de hiperparámetros (`tune_model`).
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 11

*   **Issue 31: Búsqueda de Hiperparámetros**
    *   **Descripción:** Ejecutar búsqueda avanzada de hiperparámetros (ej. con Optuna).
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 11

*   **Issue 32: Backend UI**
    *   **Descripción:** Desarrollar el backend de la UI para la carga de modelos (.pkl) e inferencia.
    *   **Asignado a:** Des. de UI
    *   **Fecha estimada:** Semana 11

*   **Issue 33: Análisis Comparativo**
    *   **Descripción:** Realizar análisis comparativo de métricas (Curvas ROC/AUC).
    *   **Asignado a:** Analista IA
    *   **Fecha estimada:** Semana 11

*   **Issue 34: Entrenamiento Final**
    *   **Descripción:** Entrenar el modelo final con los mejores hiperparámetros encontrados.
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 12

*   **Issue 35: Serialización**
    *   **Descripción:** Serializar el modelo final y el pipeline completo (.pkl).
    *   **Asignado a:** Ing. ML
    *   **Fecha estimada:** Semana 12

*   **Issue 36: Entrega Avance 2**
    *   **Descripción:** Entrega de resultados finales del modelo y avance del proyecto.
    *   **Asignado a:** Todos (Hito)
    *   **Fecha estimada:** Semana 12

---

## Sprint 6: Análisis Final y Redacción (Semanas 13-14)
**Objetivo:** Realizar análisis de interpretabilidad (SHAP), ética, finalizar la UI y redactar el informe final.

*   **Issue 37: Análisis SHAP**
    *   **Descripción:** Ejecutar scripts de SHAP y generar valores de importancia.
    *   **Asignado a:** Analista IA
    *   **Fecha estimada:** Semana 13

*   **Issue 38: Visualización Global**
    *   **Descripción:** Generar visualizaciones globales de interpretabilidad (Summary Plot).
    *   **Asignado a:** Analista IA
    *   **Fecha estimada:** Semana 13

*   **Issue 39: Análisis de Casos**
    *   **Descripción:** Analizar casos individuales específicos (Waterfall Plot).
    *   **Asignado a:** Analista IA
    *   **Fecha estimada:** Semana 13

*   **Issue 40: Ética y Sesgos**
    *   **Descripción:** Realizar un análisis ético y de sesgos del modelo desarrollado.
    *   **Asignado a:** Analista IA
    *   **Fecha estimada:** Semana 13

*   **Issue 41: Frontend Final**
    *   **Descripción:** Finalizar el desarrollo del Frontend de la UI (Streamlit).
    *   **Asignado a:** Des. de UI
    *   **Fecha estimada:** Semana 13

*   **Issue 42: Redacción de Resultados**
    *   **Descripción:** Redactar la sección de resultados e interpretabilidad del informe.
    *   **Asignado a:** Documentador
    *   **Fecha estimada:** Semana 14

*   **Issue 43: Integración UI-SHAP**
    *   **Descripción:** Integrar los gráficos de interpretabilidad (SHAP) en la interfaz de usuario.
    *   **Asignado a:** Des. de UI
    *   **Fecha estimada:** Semana 14

*   **Issue 44: Borrador Informe Final**
    *   **Descripción:** Completar el borrador del informe final del proyecto.
    *   **Asignado a:** Coordinador (Hito)
    *   **Fecha estimada:** Semana 14

---

## Sprint 7: Entrega Final (Semana 15)
**Objetivo:** Pulir el producto final, preparar la presentación y entregar el proyecto.

*   **Issue 45: Limpieza de Código**
    *   **Descripción:** Limpiar el código fuente, añadir docstrings y documentación final.
    *   **Asignado a:** Líder Técnico/Todos
    *   **Fecha estimada:** Semana 15

*   **Issue 46: Diapositivas**
    *   **Descripción:** Crear diapositivas para la presentación final.
    *   **Asignado a:** Documentador/UI
    *   **Fecha estimada:** Semana 15

*   **Issue 47: Ensayo**
    *   **Descripción:** Ensayo de la presentación final en equipo.
    *   **Asignado a:** Todos
    *   **Fecha estimada:** Semana 15

*   **Issue 48: Entrega Final**
    *   **Descripción:** Entrega formal y presentación del proyecto.
    *   **Asignado a:** Todos (Hito)
    *   **Fecha estimada:** Semana 15
