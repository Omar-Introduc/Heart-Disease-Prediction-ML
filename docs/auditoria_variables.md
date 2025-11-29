# Auditoría de Variables - Proyecto de Predicción de Enfermedades Cardíacas

Esta auditoría se basa en el análisis de los archivos `USCODE22_LLCP_102523.HTML` (Codebook del BRFSS 2022) y la lista de columnas disponibles (`columns.txt`). El objetivo es categorizar las variables para su uso en el modelo de predicción de riesgo cardíaco (`CVDINFR4` - Ataque al corazón).

## Resumen de Hallazgos Críticos

1.  **Ausencia de Variables Clave:** Las variables relacionadas con la **Presión Arterial** (ej. `BPHIGH4`) y el **Colesterol** (ej. `TOLDHI2`, `CHOLCHK`) **NO están presentes** en el conjunto de datos de 2022. Esto se debe a que en el BRFSS estos módulos son "Core Rotativo" (años impares 2021, 2023) y no formaron parte del núcleo en 2022.
    *   *Impacto:* El modelo dependerá fuertemente de comorbilidades (Diabetes, etc.) y factores de estilo de vida, perdiendo dos de los predictores clínicos más fuertes.
2.  **Fuga de Información Detectada:** La variable `MICHD` (Miocardical Infarction or CHD) es una variable calculada que incluye directamente el objetivo (`CVDINFR4`). Debe ser eliminada obligatoriamente.

---

## Categorización de Variables

A continuación se detalla la recomendación para cada grupo de variables.

### 1. Variables a Descartar (Obligatorio)

Estas variables introducen ruido, fuga de datos o son irrelevantes para la predicción clínica.

*   **Metadatos y Administración:**
    *   `STATE`, `FMONTH`, `IDATE`, `IMONTH`, `IDAY`, `IYEAR`, `DISPCODE`, `SEQNO`, `PSU`.
    *   `QSTVER`, `QSTLANG` (Versión e idioma del cuestionario).
*   **Pesos y Estratificación (Weights):**
    *   `STSTR`, `STRWT`, `RAWRAKE`, `WT2RAKE`, `CLLCPWT`, `DUALUSE`, `DUALCOR`, `LLCPWT2`, `LLCPWT`.
    *   *Razón:* Son para análisis estadístico poblacional, no para modelos predictivos a nivel individuo (a menos que se usen modelos ponderados, pero suelen añadir ruido en ML estándar).
*   **Fuga de Información (Leakage):**
    *   `MICHD`: Definida como `CVDINFR4=1 OR CVDCRHD4=1`. Revela el target.
*   **Preguntas de Contacto / Hogar:**
    *   `CTELENM1`, `PVTRESD1`, `COLGHOUS`, `STATERE1`, `CELPHON1`, `LADULT1`, `COLGSEX1`, `NUMADULT`, `LANDSEX1`, `NUMMEN`, `NUMWOMEN`, `RESPSLCT`, `CTELNUM1`, `CELLFON5`, `CADULT1`, `CELLSEX1`, `PVTRESD3`, `CCLGHOUS`, `CSTATE1`, `LANDLINE`, `HHADULT`.
    *   `NUMHHOL4`, `NUMPHON4`, `CPDEMO1C`.
    *   *Razón:* Irrelevantes para la salud cardíaca.

### 2. Variables a Descartar (Recomendado / Simplificación)

Variables que son redundantes (versiones crudas vs calculadas) o tienen alta probabilidad de valores faltantes/baja relevancia.

*   **Redundantes (Usar Calculadas/Limpias):**
    *   `HEIGHT3`, `WEIGHT2`: Descartar a favor de `BMI5` (IMC calculado) y `BMI5CAT` (Categoría). El IMC captura la relación relevante.
    *   `SMOKE100`, `SMOKDAY2`, `USENOW3`, `ECIGNOW2`: Evaluar uso de `RFSMOK3` (Fumador actual - calculado) para simplificar. `SMOKDAY2` tiene dependencia estructural.
    *   `ALCDAY4`, `AVEDRNK3`, `DRNK3GE5`, `MAXDRNKS`: `ALCDAY4` es útil. `RFDRHV8` (Heavy Drinker) es una buena variable resumen. `DROCDY4_` (Tragos por día) es buena numérica.
    *   `RACE1`, `RACEG22`, `MRACE2`...: Usar `_IMPRACE` (Imputed Race) o `_RACE` (si existe, o `RACEGR3`) para evitar múltiples columnas de raza y valores faltantes. `IMPRACE` es la más completa.
    *   `AGE`, `AGEG5YR`: Usar `AGE80` (si es numérica continua top-coded) o `AGEG5YR` (Categorías de 5 años). `AGEG5YR` es estándar en BRFSS público.
*   **Módulos Específicos / Baja Relevancia Directa:**
    *   `HADMAM`, `HOWLONG`, `CERVSCRN`... (Salud Femenina): Específicas de sexo, generan Nulos en hombres.
    *   `PSATEST1`... (Salud Masculina): Específicas de sexo.
    *   `FLUSHOT7`, `PNEUVAC4`, `TETANUS1`: Vacunación. Poca causalidad directa con IAM.
    *   `HIVTST7`...: Riesgo HIV.
    *   `CAREGIV1`... (Cuidado de otros): Determinante social, pero alto missingness en no-cuidadores.
    *   `MARIJAN1`... (Marihuana): Módulo opcional en muchos estados (posible alto missingness).
    *   `ACEDEPRS`... (Experiencias Adversas Infancia): Módulo opcional.
    *   `FIREARM5`: Armas de fuego. Irrelevante clínico.

### 3. Variables a Mantener (Predictoras Candidatas)

Estas variables deben formar el núcleo del dataset de entrenamiento.

*   **Target:**
    *   `CVDINFR4` (Ever diagnosed with heart attack).
*   **Relacionadas al Corazón (Comorbilidades Fuertes):**
    *   `CVDCRHD4` (Angina/Coronary Heart Disease): **Decisión Requerida.** Es un predictor extremadamente fuerte. Si el objetivo es predecir riesgo *antes* de cualquier evento cardíaco, descartar. Si es predecir infarto (evento agudo) en población general (incluyendo enfermos crónicos), mantener. *Recomendación: Mantener como predictor de alto riesgo.*
    *   `CVDSTRK3` (Stroke/Derrame): Fuerte comorbilidad.
*   **Historial de Salud (Chronic Conditions):**
    *   `DIABETE4` (Diabetes): Crítico.
    *   `CHCKDNY2` (Enfermedad Renal): Crítico.
    *   `CHCCOPD3` (EPOC/COPD): Importante (relacionado a fumar).
    *   `HAVARTH4` (Artritis): Indicador de inflamación/movilidad.
    *   `ADDEPEV3` (Depresión): Factor de riesgo conocido.
    *   `ASTHMA3`: Asma.
    *   `CHCSCNC1`, `CHCOCNC1` (Cáncer): Comorbilidad general.
*   **Demografía:**
    *   `SEXVAR` (Sexo) o `SEX`.
    *   `AGEG5YR` (Edad).
    *   `IMPRACE` (Raza).
    *   `EDUCA` (Educación - Proxy de nivel socioeconómico).
    *   `INCOME3` (Ingresos - Proxy NSE, cuidado con valores faltantes).
    *   `MARITAL` (Estado civil - Soporte social).
    *   `VETERAN3` (Veterano).
*   **Estilo de Vida y Salud General:**
    *   `BMI5` (Índice de Masa Corporal).
    *   `EXERANY2` (Ejercicio en el último mes). `TOTINDA` (Actividad física calculada) es mejor si está disponible.
    *   `GENHLTH` (Salud general autopercibida): Predictor muy fuerte.
    *   `PHYSHLTH` (Días de mala salud física).
    *   `MENTHLTH` (Días de mala salud mental).
    *   `SLEPTIM1` (Tiempo de sueño): Importante.
*   **Acceso a Salud:**
    *   `CHECKUP1` (Tiempo desde último chequeo).
    *   `PERSDOC3` (Tiene doctor personal).
    *   `MEDCOST1` (No pudo ir al médico por costo).

### 4. Variables de COVID-19 (Evaluar)

*   `COVIDPOS`, `COVIDVA1` (Vacuna): Pueden ser confusores temporales en 2022. Se recomienda **Descartar** para un modelo generalizable a largo plazo, a menos que el estudio sea específicamente sobre impacto COVID.

## Conclusión

El modelo deberá construirse sin lecturas directas de Presión Arterial ni Colesterol. Esto eleva la importancia de:
1.  **Diabetes (`DIABETE4`)**
2.  **Obesidad (`BMI5`)**
3.  **Tabaquismo (`RFSMOK3`)**
4.  **Historial previo (`CVDSTRK3`, `CVDCRHD4`)**
5.  **Edad y Sexo**

Se recomienda generar un dataset limpio (`processed_data.parquet`) seleccionando solo las variables de la categoría "Mantener" y filtrando las filas con valores nulos excesivos en estas columnas críticas.
