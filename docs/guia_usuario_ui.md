# Guía de Usuario - Sistema de Predicción de Riesgo Cardíaco

## Introducción
Esta aplicación web, desarrollada con **Streamlit**, permite a los profesionales de la salud estimar el riesgo de enfermedad cardíaca de un paciente utilizando un modelo de **Machine Learning (XGBoost)** entrenado con datos clínicos de NHANES (2011-2020).

## Acceso
La aplicación está disponible en la URL proporcionada por el equipo de despliegue (o localmente en `http://localhost:8501`).

## Instrucciones de Uso

### 1. Ingreso de Datos
El panel lateral izquierdo permite ingresar los datos clínicos del paciente. Los campos están divididos en cuatro secciones:

#### A. Datos Personales
* **Edad:** Entre 18 y 100 años.
* **Sexo:** Masculino o Femenino.
* **Raza/Origen:** Seleccionar la categoría más apropiada.
* **Educación/Ingresos:** Nivel educativo y ratio de ingresos (PIR).

#### B. Signos Vitales
* **IMC (BMI):** Índice de Masa Corporal.
* **Presión Sistólica:** Valor en mmHg.
* **Presión Diastólica:** Valor en mmHg (Opcional).
* **Circunferencia de Cintura:** Valor en cm.

#### C. Perfil Bioquímico
* **Colesterol:** Total, LDL, Triglicéridos.
* **Glucosa/HbA1c:** Indicadores de diabetes.
* **Enzimas/Electrolitos:** ALT, AST, GGT, Sodio, Potasio.
* **Riñón:** Creatinina, Ácido Úrico, Albúmina.

#### D. Estilo de Vida
* **Fumar:** Si ha fumado más de 100 cigarrillos en su vida.
* **Alcohol:** Consumo frecuente.
* **Actividad Física:** Actividad vigorosa regular.

### 2. Interpretación de Resultados

Una vez ingresados los datos, el sistema mostrará automáticamente la predicción en el panel principal.

#### Semáforo de Riesgo
* **🟢 Bajo Riesgo:** El modelo estima una baja probabilidad de enfermedad cardíaca.
* **🔴 Alto Riesgo / Crítico:** El modelo detecta patrones asociados con enfermedad cardíaca. Se recomienda evaluación clínica exhaustiva.

#### Probabilidad
Se muestra un porcentaje (0-100%) que indica la certeza del modelo.

### 3. Explicabilidad (SHAP)

Debajo de la predicción, se muestra un **Gráfico de Cascada (Waterfall Plot)**.
* **Barras Rojas (+):** Factores que *aumentan* el riesgo del paciente (hacia la derecha).
* **Barras Azules (-):** Factores que *disminuyen* el riesgo (hacia la izquierda).
* **Interpretación:** Este gráfico explica *por qué* el modelo tomó esa decisión específica para este paciente.

## Notas Importantes
* Esta herramienta es un **apoyo a la decisión clínica** y no sustituye el diagnóstico médico.
* Los datos se procesan localmente en la sesión y no se almacenan permanentemente.
