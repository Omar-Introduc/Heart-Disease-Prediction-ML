# Auditoría de Variables - Proyecto de Predicción de Enfermedades Cardíacas (NHANES)

Esta auditoría describe el nuevo conjunto de datos basado en **NHANES 2011-2020** (National Health and Nutrition Examination Survey). A diferencia del dataset anterior (BRFSS), este conjunto prioriza **biomarcadores clínicos objetivos** (laboratorio y exámenes) sobre encuestas subjetivas.

## Diccionario de Datos: Biomarcadores y Variables Clínicas

A continuación, se detallan las variables seleccionadas para el modelo "Clínico Predictivo".

| Variable (Feature) | Tipo | Descripción Clínica / Justificación | Unidad/Rango Típico |
| :--- | :--- | :--- | :--- |
| **HeartDisease** (Target) | Binaria | **Variable Objetivo.** Historial de infarto o ataque al corazón (MCQ160E). Define si el paciente ha sufrido un evento cardíaco. | 0: No, 1: Sí |
| **SystolicBP** | Numérica (Continua) | **Presión Arterial Sistólica.** La fuerza que ejerce la sangre contra las arterias cuando el corazón late. Es el factor de riesgo #1 para hipertensión y daño arterial. | mmHg (90 - 180+) |
| **TotalCholesterol** | Numérica (Continua) | **Colesterol Total.** Suma del colesterol en sangre. Niveles altos contribuyen a la aterosclerosis (placa en arterias). Medido en laboratorio, no preguntado. | mg/dL (100 - 300+) |
| **HbA1c** | Numérica (Continua) | **Hemoglobina Glicosilada.** Promedio de azúcar en sangre de los últimos 3 meses. Es mucho más fiable que la glucosa en ayunas para diagnosticar diabetes crónica y riesgo cardiovascular. | % (4.0 - 15.0) |
| **Glucose** | Numérica (Continua) | **Glucosa en Suero.** Nivel de azúcar en el momento del examen. Indicador agudo de metabolismo, complementa a HbA1c. | mg/dL (70 - 400+) |
| **Creatinine** | Numérica (Continua) | **Creatinina Sérica.** Producto de desecho muscular filtrado por los riñones. Niveles altos indican disfunción renal, una comorbilidad crítica para el corazón (Síndrome Cardio-Renal). | mg/dL (0.5 - 5.0+) |
| **UricAcid** | Numérica (Continua) | **Ácido Úrico.** Niveles altos (hiperuricemia) están asociados con hipertensión, inflamación y mayor riesgo cardiovascular, no solo gota. | mg/dL (2.0 - 10.0+) |
| **Triglycerides** | Numérica (Continua) | **Triglicéridos.** Tipo de grasa en sangre. Elevados junto con LDL alto aumentan drásticamente el riesgo de infarto y pancreatitis. | mg/dL (50 - 500+) |
| **LDL** | Numérica (Continua) | **Lipoproteína de Baja Densidad ("Colesterol Malo").** Transporta colesterol a las arterias. Es el objetivo primario de tratamiento para prevenir infartos. | mg/dL (50 - 200+) |
| **BMI** | Numérica (Continua) | **Índice de Masa Corporal.** Medida de obesidad basada en peso y talla reales (medidos por enfermera, no reportados por paciente). | kg/m² (15 - 60) |
| **Age** | Numérica (Continua) | **Edad.** Factor de riesgo no modificable más importante. Ahora se usa el valor exacto, no rangos decenales. | Años (20 - 80) |
| **Sex** | Binaria | **Sexo Biológico.** Hombres y mujeres tienen perfiles de riesgo distintos (ej. protección estrogénica en mujeres pre-menopáusicas). | 0: Mujer, 1: Hombre |
| **Smoking** | Binaria | **Tabaquismo.** Historial de fumar (Cotilinina o reporte clínico). Daña el endotelio vascular y acelera la aterosclerosis. | 0: No, 1: Sí |
| **Alcohol** | Frecuencia | **Consumo de Alcohol.** Consumo excesivo eleva triglicéridos y presión arterial. | Categorías/Frecuencia |
| **PhysicalActivity** | Binaria/Nivel | **Actividad Física.** Nivel de sedentarismo. El ejercicio regular mejora todos los biomarcadores anteriores. | 0: Sedentario, 1: Activo |

## Cambios en el Preprocesamiento

Dado el cambio de naturaleza de las variables (de Rangos/Categorías a Valores Reales), el pipeline de preprocesamiento cambia drásticamente:

1.  **Escalamiento Obligatorio (Scaling):**
    *   Las variables tienen magnitudes muy distintas (ej. `HbA1c` ~5.0 vs `Triglycerides` ~150.0).
    *   Se requiere **StandardScaler** (Z-score) o **RobustScaler** (si hay muchos outliers en sangre) para que el modelo no se sesgue hacia las variables de mayor magnitud.
2.  **Eliminación de One-Hot Encoding masivo:**
    *   Ya no se discretiza la edad ni el BMI. Se aprovecha la información continua.
3.  **Imputación Clínica:**
    *   Los valores nulos en laboratorio (ej. LDL faltante) deben imputarse con cuidado (ej. KNN o IterativeImputer) respetando las correlaciones biológicas, no con la media simple.

## Variables Eliminadas (Legacy BRFSS)

Se han eliminado las variables subjetivas que introducían ruido o dependían de la memoria del paciente:
*   `PhysicalHealth`, `MentalHealth` (Días de mala salud percibida).
*   `DiffWalking` (Dificultad para caminar).
*   `AgeCategory` (Reemplazado por `Age` real).
*   `GenHealth` (Salud general "Buena/Mala").
*   `SleepTime`, `Asthma`, `KidneyDisease`, `SkinCancer` (Reemplazados por sus marcadores biológicos o descartados por baja especificidad).
