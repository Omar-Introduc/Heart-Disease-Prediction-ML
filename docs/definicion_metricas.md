# Definición de Métricas de Éxito y Evaluación

**Contexto:** Predicción de Riesgo de Enfermedad Cardíaca (Clasificación Binaria)

En el ámbito médico, no todos los errores tienen el mismo peso. Clasificar incorrectamente a un paciente enfermo como sano (Falso Negativo) puede tener consecuencias fatales, mientras que clasificar a un sano como enfermo (Falso Positivo) conlleva costos de ansiedad y pruebas adicionales, pero no riesgo vital inmediato.

Por esta razón, la definición de métricas debe priorizar la **Sensibilidad (Recall)**.

## 1. Métricas Primarias (Prioridad Alta)

### 1.1. Sensibilidad (Recall / Tasa de Verdaderos Positivos)
*   **Fórmula:** $TP / (TP + FN)$
*   **Justificación:** Mide la capacidad del modelo para detectar correctamente a los pacientes que *sí* tienen la enfermedad. Queremos maximizar esto para no dejar pasar casos de riesgo.
*   **Objetivo:** > 0.90 (Idealmente). Se tolera un mayor número de Falsos Positivos a cambio de capturar la mayoría de los positivos reales.

### 1.2. F2-Score
*   **Fórmula:** $(1 + \beta^2) \cdot \frac{Precision \cdot Recall}{(\beta^2 \cdot Precision) + Recall}$ con $\beta = 2$
*   **Justificación:** El F1-Score da igual peso a la Precisión y al Recall. El **F2-Score** pondera el Recall el doble que la Precisión, alineándose mejor con la necesidad clínica de minimizar los Falsos Negativos.

## 2. Métricas Secundarias

### 2.1. AUPRC (Area Under the Precision-Recall Curve)
*   **Justificación:** Dado que los datasets médicos suelen estar desbalanceados (muchos más sanos que enfermos), el AUC-ROC puede ser optimista. El AUPRC es más robusto ante el desbalance de clases y se enfoca en la clase minoritaria (enfermos).

### 2.2. Especificidad (Tasa de Verdaderos Negativos)
*   **Fórmula:** $TN / (TN + FP)$
*   **Justificación:** Mide la capacidad de identificar correctamente a los sanos. Aunque es secundaria al Recall, no debe ser extremadamente baja para evitar saturar el sistema de salud con falsas alarmas.

## 3. Matriz de Confusión y Costos

Se evaluará el modelo considerando una matriz de costos asimétrica:

| | Predicción: Sano (0) | Predicción: Enfermo (1) |
| :--- | :--- | :--- |
| **Real: Sano (0)** | Costo: 0 | Costo: Bajo (Alarma Falsa) |
| **Real: Enfermo (1)** | **Costo: Muy Alto (Riesgo Vital)** | Costo: 0 (Detección Correcta) |

**Umbral de Decisión:**
No se utilizará el umbral por defecto de 0.5. Se realizará un análisis de curvas ROC/PR para ajustar el umbral de probabilidad de tal manera que se maximice el Recall sin degradar la precisión a niveles inaceptables.

## 4. Criterios de Éxito del Proyecto
El prototipo se considerará exitoso si logra:
1.  **Recall >= 85%** en el conjunto de prueba.
2.  **AUC-ROC >= 0.80**.
3.  Estabilidad en validación cruzada (desviación estándar baja).
