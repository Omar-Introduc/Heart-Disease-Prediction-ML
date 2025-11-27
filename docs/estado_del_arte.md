# Estado del Arte y Revisión Bibliográfica

Este documento resume los recursos clave y hallazgos del estado del arte para el proyecto, con énfasis en aplicaciones médicas, XGBoost y manejo de desbalance de clases.

## 1. Implementación Teórica de XGBoost
| Categoría | Recurso Sugerido | Descripción / Por qué leerlo |
| :--- | :--- | :--- |
| 🚀 **Paper Original** | **XGBoost: A Scalable Tree Boosting System** <br> (*T. Chen, C. Guestrin*) <br> [Enlace](https://arxiv.org/pdf/1603.02754.pdf) | Documento canónico. Explica la formulación matemática, el algoritmo de división y las optimizaciones del sistema. Esencial para un entendimiento profundo. |
| **Fundamentos** | **A Gentle Introduction to Gradient Boosting** <br> (*Machine Learning Mastery*) <br> [Enlace](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/) | Explica la intuición detrás del Gradient Boosting de forma accesible, aclarando cómo se corrigen los errores de los modelos anteriores. |
| **Fundamentos** | **Gradient Boosting Explained** <br> (*Artículo visual interactivo*) <br> [Enlace](http://explained.ai/gradient-boosting/index.html) | Una de las mejores explicaciones visuales. Permite jugar con los parámetros para ver su efecto en tiempo real. |

## 2. Aplicaciones en Salud y Cardiología
| Categoría | Recurso Sugerido (Paper) | Descripción / Aporte al Proyecto |
| :--- | :--- | :--- |
| 🩺 **Aplicación Directa** | **An Explainable Artificial Intelligence (XAI) Methodology for Heart Disease Classification** <br> (*O. M. Yaseen & M. M. Rashid, 2025*) | **Modelo a seguir.** Aplica XGBoost para clasificación cardíaca integrando directamente XAI con **SHAP y LIME**. Muestra cómo conectar la predicción con la interpretación clínica. |
| 📊 **Comparativa** | **Comparative Study of Machine Learning Algorithms in Detecting Cardiovascular Diseases** <br> (*Dayana K et al.*) <br> [Enlace](https://arxiv.org/pdf/2405.17059)| Compara XGBoost con Regresión Logística y Random Forest. Sirve de plantilla para nuestra fase de benchmarking y justificación de elección de modelo. |
| 🧠 **XAI Avanzado** | **Explainable SHAP-XGBoost models for in-hospital mortality...** <br> (*C. Tarabanis et al., 2023*) | Se enfoca en la interpretabilidad de XGBoost usando SHAP. Guía perfecta para generar y explicar gráficos de importancia, dependencia e interacción. |

## 3. Manejo de Desbalance de Clases (Imbalance Handling)

Dado que los datasets médicos (como BRFSS) suelen tener muchos más casos negativos (sanos) que positivos (enfermos), el manejo del desbalance es crítico.

### Estrategias Identificadas en la Literatura:

1.  **Algorithmic Level (Dentro de XGBoost):**
    *   **`scale_pos_weight`:** Es la técnica más recomendada y eficiente para XGBoost.
        *   *Fórmula:* `sum(negative instances) / sum(positive instances)`
        *   *Efecto:* Modifica el cálculo del gradiente para penalizar más los errores en la clase minoritaria (positiva). Es computacionalmente más barato que el re-sampling.
    *   **Referencia:** La documentación oficial de XGBoost recomienda esto sobre SMOTE para rendimiento puro en árboles.

2.  **Data Level (Re-sampling):**
    *   **SMOTE (Synthetic Minority Over-sampling Technique):** Genera ejemplos sintéticos de la clase minoritaria interpolando entre vecinos cercanos.
        *   *Pros:* Aumenta la variedad de datos de entrenamiento.
        *   *Contras:* Puede introducir ruido y aumentar el tiempo de entrenamiento.
    *   **Random Undersampling:** Eliminar aleatoriamente ejemplos de la clase mayoritaria.
        *   *Uso:* Útil si el dataset es masivo (millones de filas) para reducir carga computacional, pero se pierde información.

### Recomendación para el Proyecto:
Priorizar el uso de **`scale_pos_weight`** como primera opción debido a su integración nativa con XGBoost y eficiencia. Explorar **SMOTE** solo si el rendimiento de recall es insuficiente con la ponderación de pesos.

## 4. Ingeniería de Características y Benchmarking
| Categoría | Recurso | Aporte |
| :--- | :--- | :--- |
| 🧬 **Selección** | **Optimized Ensemble Learning Approach with Explainable AI...** <br> (*I. D. Mienye & N. Jere, 2024*) | Guía práctica para aplicar SHAP *después* del modelo optimizado para justificar qué variables son relevantes. |
| 🔬 **Benchmark** | **Cardiovascular disease risk prediction using automated machine learning** <br> (*A. Alaa et al., 2019*) | Estudio en UK Biobank. Útil para identificar predictores no tradicionales y establecer un estándar de rendimiento alto (AUC > 0.85). |
