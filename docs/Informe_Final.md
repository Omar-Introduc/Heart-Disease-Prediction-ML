# Informe Final del Proyecto: Predicción de Enfermedades Cardíacas con XGBoost

## 1. Introducción
Las enfermedades cardiovasculares (ECV) son la principal causa de muerte a nivel mundial. La detección temprana es crucial para la prevención. Este proyecto desarrolla un sistema de Machine Learning basado en **XGBoost** para predecir el riesgo de ECV utilizando datos clínicos de la encuesta **NHANES (2011-2020)**.

El objetivo principal fue construir un modelo robusto, interpretable y éticamente auditado, pasando desde una implementación "desde cero" con fines académicos hasta un despliegue productivo.

## 2. Estado del Arte
Se revisaron diversas arquitecturas, incluyendo Regresión Logística, Random Forest y Redes Neuronales. Se seleccionó **XGBoost (Extreme Gradient Boosting)** por su:
* Rendimiento superior en datos tabulares estructurados.
* Capacidad de manejo de valores nulos.
* Interpretabilidad mediante valores SHAP.
* Eficiencia computacional.

## 3. Metodología

### 3.1 Datos (NHANES)
Se consolidaron datos de 4 ciclos de NHANES (2011-2020).
* **Target:** 'HeartDisease' (Basado en autoreporte médico).
* **Features:** 20+ variables clínicas incluyendo perfil lipídico, glucosa, presión arterial, enzimas hepáticas y demografía.
* **Preprocesamiento:** Limpieza, imputación y estandarización (StandardScaler).

### 3.2 Implementación "Desde Cero" (Scratch)
Para comprender los fundamentos matemáticos, se implementó XGBoost sin librerías externas de ML:
* **Árbol de Decisión:** Cálculo de ganancia estructural basado en Gradientes ($g$) y Hessianos ($h$).
* **Función de Pérdida:** LogLoss (Entropía Cruzada Binaria).
* **Boosting:** Entrenamiento secuencial donde cada árbol corrige los errores residuales del anterior.
* **Resultado:** Se logró un prototipo funcional validado en `notebooks/05.5_XGBoost_Scratch_Demo.ipynb`.

### 3.3 Implementación Productiva (PyCaret)
Para el modelo final desplegable, se utilizó **PyCaret**:
* **Selección de Modelos:** Comparación de XGBoost, LightGBM, CatBoost.
* **Optimización:** Búsqueda de hiperparámetros enfocada en **Recall (Sensibilidad)** para minimizar Falsos Negativos.
* **Pipeline:** Preprocesamiento + Modelo serializado en `models/best_pipeline.pkl`.

## 4. Resultados Finales
El modelo final (XGBoost optimizado) alcanzó las siguientes métricas en el conjunto de prueba:
* **Recall:** Priorizado para detectar la mayor cantidad de casos positivos.
* **AUC-ROC:** Indica una buena capacidad de discriminación entre clases.
* *(Ver `notebooks/03_Model_Evaluation.ipynb` para métricas detalladas).*

## 5. Análisis de Ética y Sesgos
Se realizó una auditoría de equidad (`notebooks/06_Ethics_Analysis.ipynb`) utilizando **Fairlearn**:
* **Grupos Analizados:** Sexo y Edad.
* **Hallazgos:** Se evaluaron las Tasas de Falsos Negativos (FNR) y Positivos (FPR).
* **Conclusión:** Es fundamental monitorear la disparidad en FNR para evitar que ciertos grupos demográficos reciban diagnósticos erróneamente "sanos" con mayor frecuencia.

## 6. Despliegue y Reproducibilidad
* **Interfaz de Usuario:** Aplicación web interactiva desarrollada en **Streamlit**.
* **Contenerización:** `Dockerfile` creado para asegurar la reproducibilidad del entorno en cualquier infraestructura.
* **Documentación:** Guías de usuario y documentación técnica completa.

## 7. Conclusiones
El proyecto logró exitosamente:
1.  Demostrar la viabilidad técnica de XGBoost para predicción clínica.
2.  Implementar una versión educativa "desde cero".
3.  Desplegar una herramienta útil para profesionales de la salud.
4.  Auditar éticamente el modelo para asegurar un uso responsable.

---
**Equipo de Desarrollo**
*Sprint 7 - Entrega Final*
