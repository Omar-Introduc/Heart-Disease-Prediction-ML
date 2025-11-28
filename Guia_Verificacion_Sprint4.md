# Guía de Verificación - Sprint 4

Esta guía detalla los pasos exactos que debes seguir para auditar y verificar el funcionamiento del proyecto hasta el Sprint 4.

## 1. Verificación de Entorno y Dependencias

Antes de ejecutar cualquier código, asegúrate de tener las librerías instaladas.

**Acción:**
Ejecuta el siguiente comando en tu terminal para instalar las dependencias:
```bash
pip install -r requirements.txt
```
*Nota: Si tienes problemas con `pycaret` y Python 3.12, verifica que estés usando una versión compatible (Python 3.10 o 3.11 es recomendado).*

## 2. Verificación del Pipeline de Datos

El objetivo es asegurar que el script de ingestión puede leer el archivo crudo `.xpt` y generar el archivo procesado `.parquet`.

**Archivos Clave:**
- Entrada: `data/01_raw/LLCP2022_10rows.xpt` (Debe existir)
- Script: `src/data_ingestion.py`
- Salida Esperada: `data/02_intermediate/processed_data.parquet`

**Pasos de Prueba:**
1.  Verifica que el archivo de entrada exista.
2.  Ejecuta el script de ingestión:
    ```bash
    python src/data_ingestion.py
    ```
3.  **Criterio de Éxito:** El comando no debe mostrar errores y debe aparecer el archivo `processed_data.parquet` en la carpeta `data/02_intermediate`.

## 3. Verificación del Entrenamiento (PyCaret)

El objetivo es confirmar que PyCaret puede entrenar modelos base y guardar el mejor pipeline.

**Archivos Clave:**
- Entrada: `data/02_intermediate/processed_data.parquet`
- Script: `src/train_pycaret.py`
- Salida Esperada: `models/best_pipeline.pkl`

**Pasos de Prueba:**
1.  Ejecuta el script de entrenamiento:
    ```bash
    python src/train_pycaret.py
    ```
2.  **Criterio de Éxito:** Verás una tabla de comparación de modelos en la terminal. Al finalizar, debe existir el archivo `models/best_pipeline.pkl`.

## 4. Verificación de la Interfaz de Usuario (Streamlit)

El objetivo es validar que la aplicación web levanta correctamente y puede cargar el modelo entrenado.

**Archivos Clave:**
- Script: `src/app.py`
- Modelo Requerido: `models/best_pipeline.pkl`

**Pasos de Prueba:**
1.  Ejecuta la aplicación:
    ```bash
    streamlit run src/app.py
    ```
2.  Se abrirá una pestaña en tu navegador.
3.  **Prueba Funcional:**
    - Introduce valores en el formulario (Edad, BMI, etc.).
    - Haz clic en "Predict Risk".
    - **Criterio de Éxito:** La aplicación debe mostrar una probabilidad de riesgo (ej. "Risk Probability: 15.20%") y una alerta de "Low Risk" o "High Risk".

## 5. Auditoría Completa (Notebook)

Para una verificación técnica detallada y documentada, puedes ejecutar el notebook de auditoría.

**Archivo:** `notebooks/03_audit_sprint4.ipynb`

**Pasos:**
1.  Abre el notebook en VS Code o Jupyter.
2.  Ejecuta todas las celdas secuencialmente.
3.  **Criterio de Éxito:** Todas las celdas deben ejecutarse sin errores, confirmando la carga de datos, el funcionamiento del modelo "desde cero" (XGBoostScratch), el pipeline de PyCaret y el adaptador de la UI.


# Semantic Leakage Removal
    # Removing variables that indicate diagnosis or treatment consequence
    leakage_vars = [
        'CVDASPRN', # Aspirin usage
        'ASPUNSAF', # Aspirin unsafe
        'DIABEDU',  # Diabetes education
        
        # --- AGREGAR ESTOS NUEVOS ---
        'MICHD',    # Leakage TOTAL (Incluye el target)
        'CVDCRHD4', # Leakage parcial (Diagnóstico previo muy fuerte)
        'IDATE',    # Metadatos irrelevantes (Fecha)
        'IDAY',     # Metadatos irrelevantes (Día)
        'IYEAR',    # Metadatos irrelevantes (Año, por si acaso)
        # ----------------------------
    ]