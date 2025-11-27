# Desarrollo Matemático de XGBoost: Taylor y Hessianos

**Objetivo:** Desglosar la derivación matemática que permite a XGBoost optimizar funciones de pérdida arbitrarias utilizando información de segundo orden.

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

## 5. El Peso Óptimo ($w_j^*$)

Esta es ahora una suma de funciones cuadráticas independientes para cada hoja $w_j$. Para encontrar el peso óptimo de una hoja, derivamos respecto a $w_j$ e igualamos a cero:

$$\frac{\partial Obj}{\partial w_j} = G_j + (H_j + \lambda)w_j = 0$$

Despejando $w_j$:
$$w_j^* = - \frac{G_j}{H_j + \lambda}$$

> **Conclusión Matemática:** El valor óptimo que debe predecir una hoja no se adivina, se calcula exactamente usando la suma de gradientes y hessianos de los datos que caen en ella, regularizado por $\lambda$. Esta fórmula es la base de la implementación en código del Sprint 2.
