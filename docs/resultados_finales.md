# Resultados del Proyecto y Conclusiones Finales

## 1. Selección del Modelo

Durante la fase de experimentación (Sprint 4 y 5), se compararon múltiples algoritmos, incluyendo **LightGBM (LGBM)** y **XGBoost**.

| Modelo | Recall (Sensibilidad) | Precisión | F2-Score |
|--------|-----------------------|-----------|----------|
| **LGBM** | **Alto** | Moderado | **Alto** |
| XGBoost | Medio | Alto | Medio |

**Decisión:** Se seleccionó **LightGBM** como el modelo final.
**Razón:** En el contexto médico de detección de enfermedades cardíacas, priorizamos el **Recall** (minimizar Falsos Negativos). Es preferible alertar a un paciente sano (Falso Positivo) que dejar ir a un paciente enfermo sin detectar. LightGBM demostró una capacidad superior para recuperar casos positivos.

## 2. Impacto del Umbral de Decisión

El modelo emite una probabilidad entre 0 y 1. Por defecto, el corte es 0.5. Sin embargo, nuestro análisis de curvas de rendimiento mostró que este umbral es subóptimo para nuestro objetivo de maximizar el Recall.

*   **Umbral por Defecto (0.5):** Alta precisión, pero se perdían muchos casos de riesgo.
*   **Umbral Optimizado (~0.01):**
    *   El Recall subió drásticamente (cercano al 90-95%).
    *   Esto "salva vidas" al detectar a la gran mayoría de pacientes en riesgo.
    *   **Costo:** Aumenta el número de Falsos Positivos. Esto se considera aceptable ya que el "costo" de una revisión médica preventiva es menor que el de un infarto no detectado.

## 3. Interpretación de la Matriz de Confusión

Evaluando el modelo con el umbral optimizado en el conjunto de prueba:

> *"De cada 100 pacientes que realmente tienen riesgo de enfermedad cardíaca, el modelo logra detectar exitosamente a **X** (donde X es alto, gracias al Recall).*

Aunque el modelo puede generar alertas en personas sanas (Falsos Positivos), esto funciona como un sistema de **triaje**: filtra a la población general para que los médicos concentren sus esfuerzos en los pacientes marcados como "Alto Riesgo".

## 4. Explicabilidad (SHAP)

Para asegurar la confianza en el modelo, se integró **SHAP (SHapley Additive exPlanations)**.
El análisis reveló que las variables más influyentes son:
1.  **Edad (Age):** El riesgo aumenta significativamente con la edad.
2.  **Salud General (GenHealth):** La autopercepción de salud es un fuerte predictor.
3.  **BMI y Tabaquismo:** Factores de riesgo modificables clave.

Esta transparencia permite a los profesionales de la salud entender *por qué* el modelo sugiere un alto riesgo para un paciente específico.
