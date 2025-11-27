# Estudio Teórico de XGBoost: Fundamentos y Regularización

**Objetivo:** Comprender la "caja negra" de XGBoost, centrándonos en por qué es diferente y superior al Gradient Boosting estándar.

## 1. Introducción: De Árboles a Boosting

XGBoost (eXtreme Gradient Boosting) es una implementación optimizada del algoritmo de Gradient Boosting. Su éxito radica no solo en la velocidad computacional, sino en su formulación matemática robusta que incluye **regularización** explícita para controlar el sobreajuste.

A diferencia de Random Forest (que usa Bagging y construye árboles independientes), el Boosting construye árboles de forma **secuencial**: cada nuevo árbol intenta corregir los errores (residuales) de los anteriores.

## 2. La Función Objetivo Regularizada

El corazón de XGBoost es su función objetivo. En el paso $t$, buscamos minimizar:

$$Obj^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

Donde:
*   $\sum l(...)$: Es la función de pérdida (ej: error cuadrático, log-loss) que mide qué tan bien se ajusta el modelo a los datos.
*   $\Omega(f_t)$: Es el término de **regularización**, que penaliza la complejidad del nuevo árbol $f_t$.

### 2.1. El Término de Regularización ($\Omega$)

Esta es la gran diferencia con el Gradient Boosting tradicional (GBM). XGBoost define la complejidad del árbol como:

$$\Omega(f) = \gamma T + \frac{1}{2} \lambda ||w||^2$$

Donde:
*   $T$: Es el número de hojas en el árbol.
*   $w$: Son los pesos (scores) asignados a cada hoja.
*   **$\gamma$ (Gamma - "Min Split Loss"):** Penalización por hoja adicional. Funciona como un umbral: si la ganancia de dividir un nodo no supera a $\gamma$, la división no se realiza (pruning automático).
*   **$\lambda$ (Lambda - L2 reg):** Penalización L2 sobre los pesos de las hojas. Evita que los pesos de las hojas sean muy grandes, lo que suaviza las predicciones y reduce el overfitting.

> **Intuición:** GBM estándar es equivalente a XGBoost con $\Omega = 0$. Al añadir $\gamma$ y $\lambda$, XGBoost prefiere árboles más simples (menos hojas) y predicciones menos extremas (pesos más pequeños).

## 3. Optimización: De Gradientes a la Estructura del Árbol

Dado que no podemos optimizar la función objetivo directamente en el espacio de funciones (árboles), XGBoost utiliza una **aproximación de Taylor de segundo orden** (ver documento *Desarrollo Matemático*) para evaluar rápidamente la calidad de una estructura de árbol.

La "calidad" o "score" de una estructura de árbol $q$ se calcula sumando los scores de sus hojas:

$$Obj^* = - \frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T$$

Donde $G_j$ es la suma de gradientes y $H_j$ la suma de hessianos de los datos en la hoja $j$.

### 3.1. Criterio de División (Gain)

Para construir el árbol, el algoritmo busca iterativamente la mejor división. La ganancia de dividir un nodo en izquierda (L) y derecha (R) es:

$$Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma$$

*   El primer término mide la reducción de la pérdida gracias a la división.
*   El término $-\gamma$ actúa como el costo de complejidad. Si la mejora en la pérdida no es mayor que $\gamma$, la ganancia es negativa y no se divide.

## 4. Conclusión

XGBoost no es solo "Gradient Boosting rápido". Su formulación matemática introduce controles de complejidad ($\gamma, \lambda$) directamente en el proceso de construcción del árbol (no solo como post-pruning). Esto le permite generalizar mejor en problemas con ruido o pocos datos, haciéndolo ideal para datos médicos complejos.
