# Estudio Teórico de XGBoost: Fundamentos y Regularización

**Objetivo:** Comprender la "caja negra" de XGBoost, detallando su formulación matemática y su implementación práctica en este proyecto.

## 1. Introducción: Gradient Boosting

XGBoost (eXtreme Gradient Boosting) es una técnica de aprendizaje supervisado que construye un modelo predictivo fuerte combinando múltiples modelos débiles (típicamente árboles de decisión).

A diferencia de Random Forest, que construye árboles independientes en paralelo (Bagging), XGBoost construye árboles **secuencialmente**. Cada nuevo árbol $f_t(x)$ se entrena para corregir los errores residuales de los árboles anteriores.

## 2. Función Objetivo Regularizada

La innovación clave de XGBoost es su función objetivo, que equilibra el error de predicción con la complejidad del modelo para evitar el sobreajuste (overfitting).

$$Obj^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

Donde:
1.  **$L(...)$:** Función de pérdida (Loss Function). En nuestro proyecto, para clasificación binaria, utilizamos **LogLoss** (Entropía Cruzada Binaria).
2.  **$\Omega(f_t)$:** Término de regularización que penaliza la complejidad del nuevo árbol.

### 2.1 Regularización ($\Omega$)

Definimos la complejidad de un árbol como:

$$\Omega(f) = \gamma T + \frac{1}{2} \lambda ||w||^2$$

*   **$T$:** Número de hojas en el árbol.
*   **$w$:** Pesos (scores) en las hojas.
*   **$\gamma$ (Gamma):** Costo mínimo para añadir una nueva hoja. Actúa como un mecanismo de "poda" (pruning) durante la construcción.
*   **$\lambda$ (Lambda):** Penalización L2 sobre los pesos. Suaviza las predicciones, evitando que una sola hoja tenga un peso excesivo.

> **En nuestro código:** Estos parámetros se encuentran en `src/model.py` como `self.gamma` y `self.lambda_`.

## 3. Optimización con Gradientes de Segundo Orden

Para minimizar la función objetivo de manera eficiente, XGBoost utiliza una aproximación de **Taylor de segundo orden**. Esto permite optimizar cualquier función de pérdida siempre que sea dos veces diferenciable.

La función objetivo aproximada en el paso $t$ es:

$$Obj^{(t)} \approx \sum_{i=1}^{n} [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)$$

Donde:
*   **$g_i$ (Gradiente):** Primera derivada de la pérdida. $\partial_{\hat{y}} L$
*   **$h_i$ (Hessiano):** Segunda derivada de la pérdida. $\partial^2_{\hat{y}} L$

### Implementación de LogLoss
En `src/tree/loss_functions.py`, definimos explícitamente estas derivadas para la clasificación binaria:

*   Gradiente: $g_i = p_i - y_i$
*   Hessiano: $h_i = p_i (1 - p_i)$

Donde $p_i$ es la probabilidad predicha (Sigmoide del score).

## 4. Estructura del Árbol y Pesos Óptimos

Al agrupar las instancias por hojas, podemos calcular el peso óptimo $w_j^*$ para cada hoja $j$:

$$w_j^* = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

Esta fórmula es implementada directamente en el método `_calculate_leaf_weight` de `src/tree/decision_tree.py`. Note cómo $\lambda$ en el denominador actúa como un término de suavizado: si el Hessiano es cero o muy pequeño, $\lambda$ previene la inestabilidad numérica.

## 5. Ganancia (Gain) y Selección de Cortes

Para decidir cómo dividir un nodo, XGBoost calcula la **Ganancia de Estructura**:

$$Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma$$

*   $G_L, G_R$: Suma de gradientes en hijo izquierdo/derecho.
*   $H_L, H_R$: Suma de hessianos en hijo izquierdo/derecho.

Si $Gain > 0$, la división reduce la función de pérdida más que el costo de complejidad $\gamma$, y por tanto se acepta. Este algoritmo "Exact Greedy" recorre todos los posibles umbrales de división para encontrar el mejor.

## 6. Conclusión

La superioridad de XGBoost en competiciones y aplicaciones médicas como esta radica en:
1.  Uso de información de segundo orden (Hessiano) para una convergencia más rápida.
2.  Regularización intrínseca ($\gamma, \lambda$) que mejora la generalización.
3.  Manejo nativo de datos dispersos y no lineales.
