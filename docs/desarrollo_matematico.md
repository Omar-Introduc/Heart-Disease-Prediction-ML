# Desarrollo Matemático de XGBoost: Taylor y Hessianos

**Objetivo:** Desglosar la derivación matemática que permite a XGBoost optimizar funciones de pérdida arbitrarias utilizando información de segundo orden y conectar estas fórmulas con la implementación en código.

## 1. El Problema de Optimización

Queremos encontrar una función $f_t(x)$ (el nuevo árbol) que minimice la función objetivo en el paso $t$:

$$Obj^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

Aquí, $\hat{y}_i^{(t-1)}$ es una constante (la predicción acumulada hasta el paso anterior). Lo que buscamos es el incremento $f_t(x_i)$ que reduzca el error.

## 2. Aproximación de Taylor de Segundo Orden

La función de pérdida $l$ puede ser compleja (ej: Log-Loss, no tiene una solución cerrada fácil para árboles). Para resolver esto, XGBoost utiliza la **Expansión de Taylor** de la función de pérdida hasta el segundo orden.

Recordemos la expansión de Taylor para una función $f(x + \Delta x)$:
$$f(x + \Delta x) \approx f(x) + f'(x)\Delta x + \frac{1}{2}f''(x)\Delta x^2$$

Aplicando esto a nuestra función de pérdida, donde la "variable" actual es la predicción anterior $\hat{y}_i^{(t-1)}$ y el "incremento" $\Delta x$ es el nuevo árbol $f_t(x_i)$:

$$l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)$$

Donde definimos:
*   **Gradiente ($g_i$):** La primera derivada de la pérdida respecto a la predicción.
    $$g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$$
    *Interpretación:* La dirección en la que debemos mover la predicción para reducir el error.
*   **Hessiano ($h_i$):** La segunda derivada de la pérdida.
    $$h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$$
    *Interpretación:* La curvatura de la función de pérdida. Nos dice "qué tan rápido" cambia el gradiente.

> **Importancia del Hessiano:** El Gradient Boosting estándar (GBM) solo usa $g_i$ (descenso de gradiente). XGBoost usa $h_i$ (método de Newton), lo que le permite dar pasos más precisos hacia el mínimo, convergiendo más rápido y mejor.

## 3. Simplificación de la Función Objetivo

Sustituyendo la aproximación de Taylor en la ecuación original y eliminando los términos constantes (como $l(y_i, \hat{y}_i^{(t-1)})$ que no dependen de $f_t$), obtenemos la función objetivo aproximada:

$$Obj^{(t)} \approx \sum_{i=1}^{n} [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)$$

## 4. Agrupación por Hojas

Un árbol $f_t(x)$ asigna cada instancia $i$ a una hoja $j$. Sea $I_j = \{i | q(x_i) = j\}$ el conjunto de índices de datos que caen en la hoja $j$. Podemos reescribir la suma sobre instancias ($i$) como una suma sobre hojas ($j$):

$$Obj^{(t)} \approx \sum_{j=1}^{T} [(\sum_{i \in I_j} g_i) w_j + \frac{1}{2} (\sum_{i \in I_j} h_i + \lambda) w_j^2] + \gamma T$$

Definimos las sumas acumuladas por hoja:
*   $G_j = \sum_{i \in I_j} g_i$
*   $H_j = \sum_{i \in I_j} h_i$

Entonces:
$$Obj^{(t)} = \sum_{j=1}^{T} [G_j w_j + \frac{1}{2} (H_j + \lambda) w_j^2] + \gamma T$$

## 5. El Peso Óptimo ($w_j^*$) y la Conexión con el Código

Esta es ahora una suma de funciones cuadráticas independientes para cada hoja $w_j$. Para encontrar el peso óptimo de una hoja, derivamos respecto a $w_j$ e igualamos a cero:

$$\frac{\partial Obj}{\partial w_j} = G_j + (H_j + \lambda)w_j = 0$$

Despejando $w_j$:
$$w_j^* = - \frac{G_j}{H_j + \lambda}$$

### Conexión con el Código
En `src/tree/decision_tree.py`, el método `_calculate_leaf_weight` implementa esta fórmula exacta:

```python
def _calculate_leaf_weight(self, g, h):
    """
    Calculates the optimal weight for a leaf: w* = - Sum(g) / (Sum(h) + lambda)
    """
    G = np.sum(g)
    H = np.sum(h)
    return -G / (H + self.lambda_)
```

El parámetro `self.lambda_` en el denominador corresponde a la regularización $\lambda$, que evita la división por cero (si $H=0$) y reduce la magnitud del peso para prevenir sobreajuste.

## 6. La Ganancia de Estructura (Score)

Si sustituimos el peso óptimo $w_j^*$ de vuelta en la función objetivo, obtenemos el valor mínimo de la pérdida para una estructura de árbol dada:

$$Obj^* = - \frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T$$

Para decidir si dividir un nodo en dos (Izquierda y Derecha), comparamos el score antes de dividir vs. después de dividir. La **Ganancia** se define como la reducción en la función de pérdida:

$$Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

### Conexión con el Código
Esta fórmula es el núcleo del método `_calculate_gain` en nuestra implementación:

```python
def _calculate_gain(self, g_L, h_L, g_R, h_R):
    # ...
    term_L = (G_L**2) / (H_L + self.lambda_)
    term_R = (G_R**2) / (H_R + self.lambda_)
    term_Total = ((G_L + G_R)**2) / (H_L + H_R + self.lambda_)

    gain = 0.5 * (term_L + term_R - term_Total) - self.gamma
    return gain
```

Aquí, `self.gamma` actúa como el umbral mínimo de mejora requerido para realizar una división (pruning).
