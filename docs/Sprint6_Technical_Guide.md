# Guía Técnica para Sprint 6: Interpretabilidad, Ética y Reporte Final

Esta guía detalla las recomendaciones técnicas y estratégicas para abordar los Issues 37-40 del Sprint 6, asegurando eficiencia computacional y profundidad analítica.

## Contexto
El desarrollo actual utiliza una muestra del 0.3% de la población (aprox. 1,000 filas) para validación rápida. Sin embargo, el código implementado debe ser robusto y escalable para el entrenamiento final con el dataset completo (400,000+ filas).

## Issues 37, 38 y 39: Interpretabilidad con SHAP

### 1. Optimización del Rendimiento (Issue 37)
**Desafío:** SHAP es computacionalmente costoso sobre el dataset completo.
**Solución:** Utilizar una muestra representativa (background data) para inicializar el explicador.

**Implementación:**
- No calcular valores SHAP sobre todo el set de entrenamiento.
- Tomar una muestra aleatoria de 100 a 500 instancias del conjunto de entrenamiento.

```python
import shap
# Suponiendo que 'final_model' es el pipeline de PyCaret
model_step = final_model.named_steps['trained_model']

# Usar una muestra para el 'background'
X_train_sample = get_config('X_train').sample(n=min(100, len(get_config('X_train'))), random_state=42)
explainer = shap.TreeExplainer(model_step)
shap_values = explainer.shap_values(X_test)
```

### 2. Integración en Streamlit (Issues 39 y 32)
**Desafío:** Los gráficos estáticos en notebooks no aportan valor al usuario final.
**Solución:** Integrar explicaciones dinámicas en la UI.

**Implementación:**
- Usar `streamlit-shap`.
- Agregar checkbox "¿Por qué este resultado?" en `src/app.py`.

```python
import streamlit_shap
from streamlit_shap import st_shap

if st.checkbox("¿Por qué este resultado?"):
    st_shap(shap.force_plot(explainer.expected_value, shap_values_single, input_df))
```

### 3. Análisis de Waterfall (Issue 39)
**Desafío:** El Summary Plot es demasiado general.
**Solución:** Seleccionar 4 casos específicos para el reporte detallado.

**Casos a Analizar:**
1.  **Verdadero Positivo (TP):** Enfermo detectado correctamente.
2.  **Verdadero Negativo (TN):** Sano detectado correctamente.
3.  **Falso Negativo (FN - CRÍTICO):** Enfermo clasificado como sano. *Prioridad Alta.*
4.  **Falso Positivo (FP):** Sano clasificado como enfermo.

## Issue 40: Ética y Sesgos

### 1. Métricas Correctas
**Desafío:** Accuracy no mide equidad.
**Solución:** Enfocarse en **Tasa de Falsos Negativos (FNR) Parity**.

**Por qué:** En medicina, un FN es más grave que un FP. Se debe verificar si el modelo falla más en ciertos grupos (ej. Mujeres vs Hombres).

### 2. Variables Protegidas
**Herramienta:** `fairlearn`.
**Variables:** Sexo (`Sex`), Edad (`Age`).

**Implementación:**

```python
from fairlearn.metrics import MetricFrame
from sklearn.metrics import recall_score

grouped_metric = MetricFrame(
    metrics=recall_score,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test['Sex']
)
print(grouped_metric.by_group)
```
*Alerta:* Si el Recall varía significativamente (ej. 0.60 vs 0.85), existe un sesgo reportable.

## Recomendaciones Generales para el Informe Final

1.  **Narrativa:** Conectar métricas con hallazgos de interpretabilidad y ética.
2.  **Entorno:** Asegurar dependencias (`shap`, `fairlearn`, `streamlit-shap`) instaladas.
3.  **No Sobreentrenar el Análisis:** Usar siempre el conjunto de Test/Hold-out para estos análisis, nunca el de entrenamiento visto por el modelo.
