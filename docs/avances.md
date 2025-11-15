# **Informe de Avance del Proyecto: Predicción de Enfermedades Cardiovasculares con XGBoost**

---

### **1.0 RESUMEN EJECUTIVO**

Este documento actualiza el estado del proyecto, marcando la **finalización exitosa de la Fase 2: Preparación y Pruebas**. Se ha completado tanto el Análisis Exploratorio de Datos (EDA) inicial como el análisis visual dirigido, proporcionando una comprensión profunda de las características del dataset BRFSS 2022. Adicionalmente, se ha optimizado el flujo de trabajo de datos para mejorar la modularidad y eficiencia, y se ha refinado la estrategia de control de versiones para gestionar archivos de gran tamaño. El proyecto avanza según lo planificado hacia la fase de implementación teórica.

---

### **2.0 ESTADO DEL PROYECTO POR FASES**

#### **2.1 Fase 1: Planificación y Fundamentos (Completada ✅)**

Esta fase estableció el marco teórico y la infraestructura del proyecto, incluyendo la revisión del estado del arte de XGBoost en cardiología y la configuración del entorno de desarrollo y repositorio Git.

#### **2.2 Fase 2: Preparación y Análisis de Datos (Completada ✅)**

El objetivo de esta fase, ahora concluida, era realizar un análisis exhaustivo de los datos de origen para informar las fases posteriores de modelado.

* **2.2.1 Selección y Carga del Dataset (Tarea `p2_data`)**: Se seleccionó y cargó el dataset BRFSS 2022, validando su integridad inicial.

* **2.2.2 EDA Inicial y Limpieza Básica (Tarea `p2_eda`)**: Se realizó un EDA automatizado con `Sweetviz` tras seleccionar 21 características de alto impacto. Los pasos clave incluyeron el filtrado de valores nulos y la binarización de la variable objetivo (`CVDINFR4`), culminando en un DataFrame de **442,067 registros**.

* **2.3.3 Optimización del Flujo de Trabajo (Workflow)**:
    * **Modularización de Notebooks**: Se ha refactorizado el flujo de trabajo. El notebook `01_exploratory_data_analysis.ipynb` ahora tiene la responsabilidad única de cargar los datos crudos (`.XPT`), aplicar la limpieza inicial y **exportar el resultado** a un archivo intermedio (`brfss_2022_cleaned.csv`) en la carpeta `data/02_processed/`.
    * **Eficiencia**: Este enfoque desacopla la preparación de datos del análisis. El notebook `02_visual_analysis.ipynb` y futuros notebooks de modelado ahora cargan directamente el archivo CSV procesado, lo que acelera drásticamente la inicialización y garantiza que todos los análisis parten de una base de datos idéntica y reproducible.

* **2.3.4 Análisis Visual Dirigido (Tarea `p2_viz`)**:
    * **Objetivo**: Profundizar en los hallazgos del EDA inicial para validar hipótesis clave mediante visualizaciones específicas con `Seaborn` y `Matplotlib`.
    * **Hallazgos Clave Visualizados**:
        * **Desbalance de Clases**: Se cuantificó y visualizó con un `countplot` el severo desbalance de la variable objetivo, confirmando que menos del 6% de la población de la encuesta reportó un infarto.
        * **Análisis Bivariado**: Se generaron `KDE plots` que demuestran una clara correlación positiva entre la edad (`_AGEG5YR`) y la incidencia de infartos. `Bar plots` normalizados revelaron que categorías como ser fumador (`_SMOKER3`) o tener diabetes (`DIABETE4`) presentan un porcentaje notablemente mayor de casos positivos. `Boxplots` mostraron diferencias en la distribución del IMC (`_BMI5`) entre pacientes con y sin historial de infarto.
        * **Análisis Multivariado**: Un `heatmap` de correlación de las variables numéricas proporcionó una visión general de las interacciones lineales, ayudando a identificar posibles problemas de multicolinealidad para modelos lineales (aunque menos crítico para XGBoost).

* **2.3.5 Refinamiento del Control de Versiones**:
    * **Gestión de Archivos Grandes**: Se ha actualizado el archivo `.gitignore` para **excluir explícitamente** los archivos de datos pesados (`/data/01_raw/LLCP2022.XPT` y `/data/02_processed/brfss_2022_cleaned.csv`).
    * **Resultado**: Esta acción mantiene el repositorio Git ligero y rápido, adhiriéndose a las mejores prácticas que dictan que el control de versiones debe gestionar el **código fuente**, no los artefactos de datos grandes y reproducibles. La estructura de carpetas se mantiene mediante el uso de archivos `.gitkeep`.

---

### **3.0 Próximos Pasos Inmediatos**

Con el análisis de datos completo, el proyecto avanza hacia la **Semana 6**, iniciando la parte más teórica de la implementación.

* **Tarea Inmediata (Fase 2 - `p2_tree`)**:
    * **Acción**: Comenzar la implementación "desde cero" de los componentes de XGBoost.
    * **Objetivo**: Iniciar con la tarea **`Codificar Árbol de Decisión Base (NumPy)`**. El propósito de esta fase no es crear una librería competitiva, sino solidificar la comprensión de los mecanismos internos del algoritmo, como el cálculo de la ganancia (`gain`) y la construcción de nodos, antes de pasar a utilizar la librería optimizada en la Fase 4.

El proyecto se encuentra en una posición excelente, con una base de datos bien comprendida y un flujo de trabajo robusto, listo para abordar los desafíos de la implementación del modelo.

---

## Fase 4: Implementación del Prototipo (Semanas 9-10) - Dónde se Realiza el Preprocesamiento Exhaustivo

### Objetivo: Construir un Pipeline Robusto y Reutilizable

Aquí es donde tu plan de proyecto, de manera muy acertada, ubica la ingeniería de datos pesada.

1.  **`p4_pipe: Codificar Pipeline de Datos (en .py)` (Tarea Crítica)**:
    * **¿Qué se hará aquí?**: Esta es la tarea donde se implementa el **preprocesamiento exhaustivo**. Se creará un script en el directorio `src/`, probablemente llamado `data_pipeline.py` o `feature_engineering.py`.
    * Este script contendrá un **pipeline de Scikit-learn** (o funciones equivalentes) que ejecutará, en orden y de forma reproducible, todas las operaciones que decidimos no hacer en la Fase 2:
        1.  **Imputación de valores faltantes** (para todas las variables predictoras).
        2.  **Codificación de variables categóricas**.
        3.  **Escalado de variables numéricas**.
        4.  **Aplicación de técnicas de muestreo** (como SMOTE) para corregir el desbalance de clases.
    * **¿Por qué aquí y no antes?**: Porque las decisiones sobre *cómo* preprocesar los datos se basan en los hallazgos del análisis de la Fase 2 y en el diseño de la arquitectura de la Fase 3. No se puede construir un pipeline robusto sin haber analizado primero a fondo los datos crudos.

**En resumen**: Tu plan sigue una metodología profesional. No estás omitiendo el preprocesamiento; lo has colocado estratégicamente en la fase de implementación (`p4_pipe`), que es exactamente donde debe estar para construir una solución de Machine Learning modular, reproducible y robusta.


### **3.1 Implementación del Árbol de Decisión Base (Issue-02.4)**

Como parte de la Fase 2 y en línea con el objetivo de comprender a fondo los componentes de XGBoost, se ha completado la implementación de un Árbol de Decisión desde cero utilizando NumPy.

*   **Propósito de la Implementación**:
    *   **Entendimiento Fundamental**: El objetivo principal no es reemplazar la librería optimizada de XGBoost, sino construir una comprensión profunda de los mecanismos internos que impulsan los modelos basados en árboles.
    *   **Componentes Clave Desarrollados**:
        *   **Estructura de Nodos**: Se creó una clase `Node` que sirve como el bloque de construcción fundamental del árbol.
        *   **Lógica de División (Splitting)**: Se implementaron funciones para encontrar la mejor división en los datos, utilizando criterios de impureza para maximizar la ganancia de información.
        *   **Cálculo de Impureza**: El árbol soporta tanto la **Entropía** como el **Índice de Gini**, permitiendo flexibilidad en la evaluación de las divisiones.
        *   **Predicción y Recorrido**: Se desarrollaron métodos para atravesar el árbol y realizar predicciones sobre nuevos datos.
    *   **Ubicación del Código**: El código fuente de esta implementación se encuentra en el nuevo directorio `src/tree/`, asegurando una estructura de proyecto modular y organizada.

Esta implementación sirve como una base teórica y práctica crucial antes de avanzar hacia el ensamblaje de árboles más complejo que define a XGBoost.
