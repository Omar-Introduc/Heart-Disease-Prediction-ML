# Propuesta de Proyecto: Sistema de Machine Learning para la Predicción de Riesgo de Enfermedad Cardíaca a partir de Datos Clínicos

**Curso:** Inteligencia Artificial
**Profesor:** Marcos Antonio
**Integrantes:**
* Jhiens
* Joel
* Luiggi
* Emhir
* Hermoza

**Lima - Perú**
**Septiembre 2025**

---

## 1. Descripción Inicial del Problema

### 1.1. Contexto del Problema
Las enfermedades cardiovasculares constituyen actualmente la principal causa de mortalidad a nivel mundial, con una cifra estimada de 17.9 millones de muertes anuales según la Organización Mundial de la Salud (OMS) [1]. Este grupo de patologías no solo impacta en la expectativa y calidad de vida de millones de personas, sino que también genera una elevada carga económica y social en los sistemas de salud [2, 3, 4].

Un aspecto crítico es que gran parte de los diagnósticos se realizan en etapas avanzadas, cuando el paciente ya presenta síntomas graves o complicaciones. Esto se debe a que la evaluación del riesgo cardiovascular suele depender en gran medida de la experiencia clínica del profesional y de métodos tradicionales que no siempre logran capturar patrones complejos en los datos [5, 7].

En este contexto, surge la necesidad de implementar herramientas basadas en inteligencia artificial que permitan analizar grandes volúmenes de información clínica de manera sistemática y precisa. La detección temprana del riesgo cardíaco, apoyada en técnicas de machine learning, no solo puede mejorar la capacidad predictiva y la toma de decisiones médicas, sino también optimizar el uso de recursos, facilitar la prevención y contribuir a reducir la tasa de mortalidad asociada a estas enfermedades [6].

### 1.2. Problemática Específica
El diagnóstico de enfermedades cardiovasculares presenta varios desafíos que dificultan una detección oportuna y precisa:
1.  **Diagnóstico tardío:** Una proporción considerable de pacientes es identificada como de alto riesgo solo cuando aparecen síntomas graves o complicaciones, lo que reduce las posibilidades de intervención temprana y preventiva [5, 4].
2.  **Evaluación subjetiva:** Los métodos tradicionales dependen en gran medida de la experiencia y criterio clínico individual, lo que puede generar variabilidad en los resultados y decisiones médicas [1, 2].
3.  **Manejo de grandes volúmenes de datos:** Los profesionales de la salud deben procesar información clínica extensa y heterogénea (historiales médicos, exámenes de laboratorio, imágenes), lo que dificulta un análisis sistemático sin el apoyo de herramientas avanzadas [6].
4.  **Limitación de recursos:** En muchos sistemas de salud, el tiempo y la disponibilidad de especialistas son reducidos, lo que incrementa la necesidad de sistemas de apoyo al diagnóstico que permitan optimizar los recursos existentes y mejorar la cobertura de atención [2, 3].

### 1.3. Justificación
El desarrollo de un sistema de predicción de riesgo de enfermedad cardíaca mediante técnicas de inteligencia artificial se justifica por múltiples razones:
1.  **Impacto en la salud pública:** Las enfermedades cardiovasculares constituyen la principal causa de mortalidad global, por lo que cualquier avance en su detección temprana tiene un efecto directo en la reducción de muertes y complicaciones asociadas [1, 3].
2.  **Mejora en la precisión diagnóstica:** Los modelos de machine learning pueden identificar patrones complejos en grandes volúmenes de datos clínicos que resultan difíciles de detectar con métodos tradicionales, contribuyendo a diagnósticos más objetivos y consistentes [6, 4].
3.  **Optimización de recursos médicos:** La implementación de sistemas automatizados de apoyo a la decisión clínica permite un uso más eficiente del tiempo de los profesionales de la salud, facilitando la priorización de pacientes de alto riesgo y reduciendo costos asociados [2].
4.  **Segunda opinión confiable:** Los sistemas predictivos no buscan reemplazar al médico, sino ofrecer una herramienta de apoyo que incremente la seguridad en la toma de decisiones clínicas, complementando la experiencia profesional con evidencia basada en datos [5].
5.  **Contribución académica y tecnológica:** El proyecto fortalece la aplicación de la inteligencia artificial en el ámbito biomédico, promoviendo la investigación interdisciplinaria y aportando soluciones innovadoras en salud digital.

---

## 2. Objetivos

### 2.1. Objetivo General
Desarrollar un sistema predictivo basado en técnicas de machine learning que permita estimar el riesgo de enfermedad cardíaca en pacientes a partir de datos clínicos y biomédicos, con el fin de apoyar la toma de decisiones médicas, contribuir a la detección temprana y optimizar los recursos en salud.

### 2.2. Objetivos Específicos
1.  Recopilar y analizar datasets relevantes de enfermedades cardiovasculares, verificando su calidad y pertinencia para el problema de predicción.
2.  Preprocesar y normalizar los datos clínicos, aplicando técnicas de limpieza, imputación de valores faltantes y selección de características relevantes.
3.  Implementar y comparar diferentes algoritmos de clasificación supervisada (Regresión Logística, Random Forest, SVM, XGBoost, Redes Neuronales), evaluando su rendimiento y robustez.
4.  Optimizar y validar los modelos mediante técnicas de validación cruzada y ajuste de hiperparámetros, empleando métricas específicas (Accuracy, Recall, F1-score, AUC-ROC).
5.  Garantizar interpretabilidad, aplicando herramientas como SHAP y LIME para facilitar la comprensión de los resultados por parte de los profesionales de la salud.
6.  Diseñar una interfaz de usuario que permita la interacción sencilla con el sistema predictivo y documentar el proceso para su futura implementación y mantenimiento.

---

## 3. Identificación de Técnicas de IA a Utilizar

### 3.1. Algoritmos de Machine Learning Supervisado
Se emplearán diferentes modelos de clasificación binaria y multiclase para predecir el riesgo de enfermedad cardíaca, con el fin de comparar desempeño y robustez:
* **Regresión Logística:** Modelo base por su simplicidad y capacidad interpretativa.
* **Support Vector Machines (SVM):** Adecuado para clasificaciones no lineales con kernels como RBF.
* **Random Forest y Gradient Boosting (XGBoost/LightGBM):** Técnicas de ensamble que permiten alta precisión, buen manejo de datos mixtos y análisis de importancia de variables.
* **Redes Neuronales (MLP y DNN):** Útiles para capturar patrones complejos y relaciones no lineales en los datos.

### 3.2. Técnicas de Preprocesamiento
Para garantizar la calidad y utilidad del dataset se aplicarán:
* **Normalización y estandarización:** Uso de MinMaxScaler y StandardScaler para homogeneizar variables.
* **Codificación de variables categóricas:** One-Hot Encoding y Label Encoding.
* **Imputación de valores faltantes:** Métodos como KNN Imputer e Iterative Imputer.
* **Selección de características:** Eliminación de redundancias mediante correlación de Pearson, Recursive Feature Elimination (RFE) y técnicas de reducción dimensional como PCA.

### 3.3. Técnicas de Evaluación y Validación
La validez y desempeño de los modelos se comprobarán mediante:
* **Validación cruzada (K-Fold, Stratified K-Fold):** Para evaluar estabilidad del modelo en diferentes particiones.
* **Hold-out validation:** División en conjuntos train/validation/test.
* **Métricas específicas:** Accuracy, Precision, Recall, F1-Score, AUC-ROC, Matriz de Confusión, Sensibilidad y Especificidad (críticas en contextos médicos).

### 3.4. Técnicas de Interpretabilidad
Con el fin de garantizar transparencia en los resultados y confianza clínica, se aplicarán:
* **SHAP (SHapley Additive exPlanations):** Para estimar la contribución de cada característica en la predicción.
* **LIME (Local Interpretable Model-agnostic Explanations):** Para interpretaciones locales de casos individuales.
* **Gráficos de importancia de variables y Partial Dependence Plots:** Para facilitar la comprensión de relaciones entre características y riesgo.

### 3.5. Herramientas y Frameworks
El proyecto se desarrollará en un entorno Python utilizando:
* **Scikit-learn, Pandas y NumPy:** Para preprocesamiento, modelado y análisis de datos.
* **TensorFlow/Keras:** Para el desarrollo de redes neuronales.
* **Matplotlib y Seaborn:** Para visualización gráfica.
* **Streamlit/Flask:** Para la implementación de una interfaz de usuario interactiva y de fácil uso por profesionales de la salud.