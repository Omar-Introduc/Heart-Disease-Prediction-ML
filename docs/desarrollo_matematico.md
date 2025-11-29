# Desarrollo Matemático de XGBoost: Derivación y Fórmulas

**Objetivo:** Detallar paso a paso la derivación matemática que conecta la función de pérdida LogLoss con el algoritmo de construcción de árboles implementado en `src/`.

## 1. Función de Pérdida (LogLoss)

Para un problema de clasificación binaria, queremos minimizar la entropía cruzada. Dado un label real $y_i \in \{0, 1\}$ y una predicción de probabilidad $p_i$, la pérdida es:

$$L(y_i, p_i) = -[y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$

XGBoost trabaja con "log-odds" (logits), denotados como $\hat{y}$. La probabilidad se obtiene mediante la función Sigmoide:
$$p_i = \sigma(\hat{y}_i) = \frac{1}{1 + e^{-\hat{y}_i}}$$

## 2. Derivadas: Gradiente y Hessiano

XGBoost requiere las derivadas respecto al score $\hat{y}$.

### Gradiente ($g_i$) - Primera Derivada
$$\frac{\partial L}{\partial \hat{y}} = p_i - y_i$$

*Demostración rápida:* $\frac{\partial L}{\partial p} = \frac{p-y}{p(1-p)}$ y $\frac{\partial p}{\partial \hat{y}} = p(1-p)$. Al multiplicar (Regla de la Cadena), los términos $p(1-p)$ se cancelan.

**En Código (`src/tree/loss_functions.py`):**
```python
def gradient(self, y_true, y_pred_score):
    p = self.sigmoid(y_pred_score)
    return p - y_true
```

### Hessiano ($h_i$) - Segunda Derivada
$$\frac{\partial^2 L}{\partial \hat{y}^2} = \frac{\partial (p_i - y_i)}{\partial \hat{y}} = \frac{\partial p_i}{\partial \hat{y}} = p_i(1 - p_i)$$

**En Código (`src/tree/loss_functions.py`):**
```python
def hessian(self, y_true, y_pred_score):
    p = self.sigmoid(y_pred_score)
    return p * (1 - p)
```

## 3. Aproximación de Taylor

Queremos encontrar el incremento $f_t(x)$ que minimice la pérdida. Aproximamos la función alrededor de la predicción actual $\hat{y}^{(t-1)}$:

$$L(y, \hat{y}^{(t-1)} + f_t(x)) \approx L(y, \hat{y}^{(t-1)}) + g_i f_t(x) + \frac{1}{2} h_i f_t(x)^2$$

Eliminando los términos constantes, obtenemos la función objetivo simplificada que solo depende de $f_t(x)$:

$$Obj \approx \sum_{i=1}^n [g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2] + \Omega(f_t)$$

## 4. Solución Exacta para los Pesos ($w$)

Reescribiendo la suma sobre las hojas $j=1...T$:

$$Obj = \sum_{j=1}^T [ (\sum_{i \in I_j} g_i) w_j + \frac{1}{2} (\sum_{i \in I_j} h_i + \lambda) w_j^2 ] + \gamma T$$

Sea $G_j = \sum g_i$ y $H_j = \sum h_i$. Para encontrar el mínimo, derivamos respecto a $w_j$ e igualamos a 0:

$$G_j + (H_j + \lambda) w_j = 0$$

$$w_j^* = \frac{-G_j}{H_j + \lambda}$$

Este es el valor que se asigna a cada hoja del árbol.

## 5. Puntuación de Estructura (Score)

Sustituyendo $w_j^*$ en la ecuación de $Obj$, obtenemos el valor óptimo de la función objetivo para una estructura dada:

$$Obj^* = -\frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j + \lambda} + \gamma T$$

Cuanto menor sea este valor, mejor es la estructura.

## 6. Ganancia de División (Gain)

Para evaluar si dividir un nodo es beneficioso, comparamos el score del nodo padre (sin dividir) contra la suma de los scores de los hijos (Izquierdo + Derecho).

$$Gain = \text{Score}_{Padre} - (\text{Score}_{Izq} + \text{Score}_{Der})$$
*(Nota: XGBoost minimiza Obj, pero maximiza Gain. La fórmula se invierte en signo)*.

$$Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma$$

**Interpretación:**
1.  **Primer término:** Mejora en la reducción del error gracias a la división.
2.  **$\gamma$:** Penalización por complejidad. Si la mejora no supera $\gamma$, la división no se realiza.

Esta fórmula es el núcleo de la decisión en `src/tree/decision_tree.py`.
