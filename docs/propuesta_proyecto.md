# Propuesta de Proyecto: Sistema de Predicción de Riesgo Cardiovascular basado en Biomarcadores Clínicos (NHANES)

**Curso:** Inteligencia Artificial
**Profesor:** Marcos Antonio
**Integrantes:**
* Jhiens
* Joel
* Luiggi
* Emhir
* Hermoza

**Lima - Perú**
**Septiembre 2025**

---

## 1. Descripción del Problema

### 1.1. Nuevo Enfoque: De la Encuesta a la Clínica
Las enfermedades cardiovasculares siguen siendo la principal causa de muerte global. Sin embargo, los modelos de predicción tradicionales a menudo fallan por basarse en datos subjetivos: encuestas donde el paciente "recuerda" su estado de salud.

Este proyecto marca un **cambio de paradigma**. Hemos migrado de datos de encuestas (BRFSS) a datos clínicos rigurosos del **NHANES 2011-2020** (National Health and Nutrition Examination Survey). Ya no buscamos patrones en lo que la gente *dice*, sino en lo que su sangre *revela*.

### 1.2. Problemática Específica
1.  **Subjetividad vs. Objetividad:** Un paciente puede decir que "come sano", pero sus niveles de Triglicéridos y HbA1c cuentan la historia real metabólica. Los modelos basados en encuestas tienen un techo de precisión bajo debido a este sesgo de memoria.
2.  **La "Caja Negra" Metabólica:** El riesgo cardíaco es una combinación compleja y no lineal de factores. Un nivel de glucosa "ligeramente alto" puede ser fatal si se combina con presión alta y colesterol LDL elevado, una interacción que las reglas simples médicas a veces pasan por alto.
3.  **Detección de Fenotipos Silenciosos:** Muchos pacientes son pre-diabéticos o hipertensos asintomáticos. Un modelo clínico puede detectar estos "fenotipos de riesgo" años antes de un infarto, basándose en biomarcadores sutiles.

### 1.3. Justificación
La adopción de un enfoque basado en biomarcadores (Biomarker-based approach) se justifica por:
1.  **Fiabilidad del Dato:** Usamos variables "duras" (Hard Data) como Presión Sistólica (mmHg), Hemoglobina Glicosilada (%) y Creatinina, medidas por profesionales en laboratorios móviles, eliminando el error humano del paciente.
2.  **Medicina de Precisión:** Al usar variables continuas exactas (ej. IMC de 27.5 en lugar de "Sobrepeso"), el modelo puede afinar mucho más su predicción individualizada.
3.  **Relevancia Clínica Real:** Este sistema se acerca más a una herramienta de uso hospitalario real, donde los médicos toman decisiones basadas en exámenes de laboratorio, no solo en cuestionarios.

---

## 2. Objetivos

### 2.1. Objetivo General
Desarrollar un sistema de "Screening Clínico Avanzado" que prediga el riesgo de evento cardíaco utilizando biomarcadores sanguíneos y exámenes físicos objetivos del dataset NHANES, superando la precisión de los modelos basados en encuestas.

### 2.2. Objetivos Específicos
1.  **Ingeniería de Datos Clínicos:** Integrar y limpiar tablas complejas de NHANES (Demografía, Laboratorio, Examen Físico) uniendo datos por ID de paciente (SEQN).
2.  **Manejo de Escalas:** Implementar técnicas robustas de escalamiento (StandardScaler/RobustScaler) para normalizar variables con magnitudes físicas dispares (ej. Colesterol vs. HbA1c).
3.  **Modelado No Lineal:** Utilizar XGBoost para capturar las relaciones no lineales críticas entre biomarcadores (ej. la curva de riesgo de la presión arterial no es una línea recta).
4.  **Validación Médica:** Evaluar el modelo no solo con métricas de ML (AUC-ROC), sino verificando que las variables más importantes (Feature Importance) coincidan con la literatura cardiológica (ej. Edad, Presión, Colesterol).

---

## 3. Identificación de Técnicas de IA

### 3.1. Algoritmos
*   **XGBoost (eXtreme Gradient Boosting):** Seleccionado como motor principal por su capacidad superior para manejar datos tabulares densos y capturar interacciones complejas entre variables continuas sin necesidad de asumir linealidad.

### 3.2. Preprocesamiento Avanzado
*   **Imputación Clínica:** Uso de `IterativeImputer` (MICE) para estimar valores de laboratorio faltantes basándose en otros biomarcadores correlacionados.
*   **Escalamiento:** Uso obligatorio de `StandardScaler` para llevar todas las mediciones físicas a una distribución comparable (Z-Score).

### 3.3. Evaluación
*   **Métricas:** Prioridad en **Recall (Sensibilidad)**. En medicina, es inaceptable perder un caso positivo (Falso Negativo: predecir "Sano" a alguien que tendrá un infarto).
*   **Interpretabilidad:** Uso de SHAP values para explicarle al médico *por qué* el modelo ve riesgo (ej. "Su riesgo es alto principalmente porque su HbA1c > 6.5% y su Presión Sistólica > 140").
