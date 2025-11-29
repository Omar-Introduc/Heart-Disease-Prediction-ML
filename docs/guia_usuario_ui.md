# Guía de Usuario - Sistema de Predicción de Riesgo Cardíaco

Esta guía explica cómo utilizar la interfaz de usuario (UI) basada en Streamlit para interactuar con el modelo de predicción de enfermedades cardíacas.

## Acceso a la Aplicación

Para ejecutar la aplicación localmente (si tiene el entorno configurado):
```bash
streamlit run src/app.py
```
O usando Docker:
```bash
docker run -p 8501:8501 <nombre_imagen>
```

## Estructura de la Interfaz

La aplicación se divide en una barra lateral de configuración y un área principal de entrada de datos y resultados.

### 1. Panel Lateral (Configuración)
*   Muestra información sobre el modelo cargado (versión del pipeline).
*   Permite ajustes avanzados si están habilitados (e.g., umbral de decisión).

### 2. Formulario de Entrada
El formulario está organizado en cuatro secciones lógicas para facilitar la introducción de datos clínicos:

*   **Datos Personales:** Edad, Sexo, Raza, Educación.
*   **Signos Vitales:** Presión Arterial (Sistólica), IMC (Índice de Masa Corporal), Circunferencia de Cintura.
*   **Perfil Bioquímico:** Colesterol Total, LDL, Triglicéridos, Glucosa, HbA1c, Ácido Úrico, Creatinina, Enzimas Hepáticas (ALT, AST, GGT), Electrolitos (Sodio, Potasio), Albúmina.
*   **Estilo de Vida:** Tabaquismo, Consumo de Alcohol, Actividad Física, Seguro de Salud.

### 3. Realizar Predicción
Una vez completados los campos (los valores predeterminados representan una media poblacional o un valor neutro), presione el botón **"Calcular Riesgo"**.

## Interpretación de Resultados

### Predicción de Riesgo
El sistema mostrará uno de los tres estados posibles basado en la probabilidad calculada por el modelo XGBoost:

*   🟢 **Riesgo Bajo:** El modelo estima una probabilidad baja de enfermedad cardíaca. Se sugiere mantener hábitos saludables.
*   🟡 **Riesgo Moderado:** Probabilidad intermedia. Se recomienda monitoreo.
*   🔴 **Riesgo Alto:** Probabilidad alta. Se recomienda consulta médica inmediata.

### Explicabilidad (SHAP)
Debajo del resultado, se mostrará un **Gráfico de Cascada (Waterfall Plot)** generado por SHAP.
*   **Barras Rojas:** Indican factores que *aumentan* el riesgo (empujan la probabilidad hacia 1).
*   **Barras Azules:** Indican factores que *disminuyen* el riesgo (empujan la probabilidad hacia 0).
*   La longitud de la barra representa la magnitud del impacto de esa variable específica en la decisión final del modelo.
