# Documentación de Implementación: XGBoost "Desde Cero"

Esta documentación describe la implementación personalizada del algoritmo XGBoost realizada durante el Sprint 2. El objetivo de esta implementación es puramente académico: desmitificar la "caja negra" y validar la comprensión matemática de los gradientes y hessianos.

## 1. Estructura del Código

El código se divide en dos componentes principales ubicados en `src/`:

*   **`XGBoostScratch` (`src/model.py`):** Clase orquestadora que gestiona el ensamblaje (Boosting), el cálculo de gradientes/hessianos y la actualización aditiva de las predicciones.
*   **`DecisionTree` (`src/tree/decision_tree.py`):** Implementación de un árbol de regresión individual que aprende a predecir los gradientes negativos (residuos).

## 2. Detalle de Algoritmos

### 2.1 Algoritmo "Exact Greedy"
En el método `_find_best_split` de la clase `DecisionTree`, implementamos un enfoque **Exact Greedy**.

```python
# src/tree/decision_tree.py

def _find_best_split(self, X, g, h):
    # ...
    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx]) # <--- Puntos de corte
        for threshold in thresholds:
             # Evaluar ganancia exacta para cada posible división
             # ...
```

**Funcionamiento:**
1.  Para cada variable (columna).
2.  Ordena los valores y encuentra todos los valores únicos.
3.  Prueba *cada* valor único como un posible umbral de división.
4.  Calcula la ganancia exacta usando todos los datos.

**Limitaciones Computacionales:**
Este enfoque tiene una complejidad de $O(N \cdot d \cdot \log N)$ (por el ordenamiento) o $O(N \cdot d)$ si está pre-ordenado, donde $N$ es el número de muestras y $d$ las dimensiones. Para datasets grandes o continuos (como el BMI o edad con decimales), el número de umbrales es enorme, haciendo que el entrenamiento sea extremadamente lento.

**Diferencia con XGBoost Industrial:**
La librería oficial XGBoost (que usaremos vía PyCaret) soluciona esto mediante el **Algoritmo Aproximado**:
*   **Histogramas:** Agrupa los valores continuos en "bins" discretos.
*   **Weighted Quantile Sketch:** Propuesta de puntos de corte candidatos basada en la distribución de los hessianos (pesos), asegurando que cada bin tenga un peso similar.
Esto reduce la complejidad de probar $N$ puntos a probar $K$ bins (donde $K \ll N$).

### 2.2 Cálculo de Pesos y Ganancia
La implementación traduce directamente las fórmulas derivadas de la expansión de Taylor de segundo orden.

**Peso de la Hoja ($w^*$):**
Definido en `_calculate_leaf_weight`.
$$ w^* = -\frac{\sum g_i}{\sum h_i + \lambda} $$

**Ganancia de División:**
Definida en `_calculate_gain`.
$$ Gain = \frac{1}{2} \left[ \frac{(\sum g_L)^2}{\sum h_L + \lambda} + \frac{(\sum g_R)^2}{\sum h_R + \lambda} - \frac{(\sum g)^2}{\sum h + \lambda} \right] - \gamma $$

Donde $\lambda$ (`self.lambda_`) controla la regularización L2 y $\gamma$ (`self.gamma`) es el umbral mínimo de reducción de pérdida para permitir una división (pruning).

## 3. Interfaces
A pesar de ser una implementación académica, `XGBoostScratch` expone métodos estándar `fit`, `predict` y `predict_proba` para mantener consistencia con scikit-learn y facilitar pruebas comparativas.
