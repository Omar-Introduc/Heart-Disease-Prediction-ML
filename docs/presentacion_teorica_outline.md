# Esquema de Presentación: Teoría de XGBoost

**Público:** Equipo de Desarrollo (Ingenieros, Analistas)
**Duración:** 30 minutos
**Objetivo:** Nivelar el conocimiento del equipo sobre cómo funciona XGBoost internamente antes de implementarlo.

---

## Slide 1: Título e Introducción
*   **Título:** "Desmitificando XGBoost: Matemáticas y Estructura"
*   **Presentador:** Ing. Machine Learning
*   **Mensaje Clave:** XGBoost no es magia; es una suma de árboles optimizada con cálculo de segundo orden.

## Slide 2: ¿Por qué XGBoost?
*   Comparativa rápida:
    *   **Decision Tree:** Simple, pero propenso a overfitting.
    *   **Random Forest:** Robusto (Bagging), pero a veces lento y menos preciso.
    *   **Gradient Boosting:** Preciso (Boosting), pero lento de entrenar y difícil de regularizar.
    *   **XGBoost:** Lo mejor de ambos mundos: Velocidad + Precisión + Regularización.

## Slide 3: El Concepto de Boosting (Intuición)
*   Analogía del "Equipo de Expertos":
    *   El primer árbol hace una predicción general.
    *   El segundo árbol se enfoca en corregir los errores del primero.
    *   El tercer árbol corrige los errores de la suma de los dos anteriores.
*   Resultado Final = Suma de todas las correcciones.

## Slide 4: La Función Objetivo (El "Jefe")
*   Ecuación: $Obj = Loss + Regularization$
*   Explicación visual:
    *   **Loss:** ¿Qué tan cerca estamos del dato real? (Quiero predecir bien).
    *   **Regularization:** ¿Qué tan complicado es mi árbol? (Quiero árboles simples).
*   **Importante:** XGBoost penaliza tener muchas hojas ($\gamma$) y tener pesos muy altos en las hojas ($\lambda$).

## Slide 5: El Truco Matemático (Taylor)
*   Problema: Optimizar funciones de pérdida complejas es difícil.
*   Solución: Usar la Expansión de Taylor.
*   **Gradiente (g):** Dirección del error.
*   **Hessiano (h):** Curvatura del error (aceleración).
*   *Key Takeaway:* Usar el Hessiano permite a XGBoost converger mucho más rápido que el Gradient Boosting estándar.

## Slide 6: Construyendo el Árbol (Split Finding)
*   ¿Cómo decidimos dónde cortar una rama?
*   Fórmula de Ganancia ($Gain$).
*   Explicación de los componentes:
    *   Gain = (Mejora en Izquierda + Mejora en Derecha - Costo de no dividir) - Costo de Complejidad ($\gamma$).
*   Si la ganancia < 0 (o < $\gamma$), no dividimos (Pruning).

## Slide 7: Manejo de Datos Faltantes y Desbalance
*   **Sparsity Aware:** XGBoost aprende automáticamente hacia dónde mandar los valores nulos (izquierda o derecha) basándose en qué reduce más el error.
*   **Scale Pos Weight:** Cómo ajustamos el algoritmo para darle más importancia a detectar "Enfermos" (Clase 1) en nuestro dataset desbalanceado.

## Slide 8: Conclusiones y Siguientes Pasos
*   XGBoost es poderoso por su regularización y uso de Hessianos.
*   Siguiente Sprint: Implementaremos esto "from scratch" en Python para interiorizarlo.
*   Q&A.
