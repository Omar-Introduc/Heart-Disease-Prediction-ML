import json


def update_notebook(filepath):
    with open(filepath, "r") as f:
        nb = json.load(f)

    # 1. Update Header (Cell 0)
    nb["cells"][0]["source"] = [
        "# 📊 Análisis Exploratorio de Datos Clínicos (NHANES)\n",
        "\n",
        "## 🎯 Objetivo\n",
        "Este notebook realiza un **Análisis Exploratorio de Datos (EDA)** exhaustivo sobre el dataset clínico procesado de NHANES (National Health and Nutrition Examination Survey).\n",
        "\n",
        "El objetivo principal es validar la calidad de los datos, entender las distribuciones de los nuevos biomarcadores clínicos y evaluar su poder predictivo frente a la variable objetivo: **Enfermedad Cardíaca (HeartDisease)**.\n",
        "\n",
        "## 🛠️ Herramientas Utilizadas\n",
        "- **Pandas**: Para manipulación y estructuración de datos.\n",
        "- **Sweetviz**: Para generación automática de reportes visuales comparativos.\n",
        "- **Matplotlib/Seaborn**: Para análisis de correlaciones y visualizaciones específicas.\n",
        "\n",
        "## 📋 Flujo de Trabajo\n",
        "1. **Carga de Datos**: Importar el dataset `process_data.parquet`.\n",
        "2. **Estandarización**: Renombrar variables al inglés estándar médico.\n",
        "3. **Análisis Estadístico**: Validar rangos, promedios y desviaciones.\n",
        "4. **Correlación**: Identificar multicolinealidad entre variables.\n",
        "5. **Reporte Automático**: Generar HTML con `Sweetviz` para análisis visual profundo.",
    ]

    # 2. Update "Carga y Preparación de Datos" (Cell 3 in original, but index 3)
    # Finding the cell that starts with "## 1. Carga..."
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown" and cell["source"][0].startswith(
            "## 1. Carga"
        ):
            cell["source"] = [
                "## 1. Carga y Preparación de Datos\n",
                "\n",
                "En esta sección cargamos los datos procesados. El formato **Parquet** se utiliza por su eficiencia en lectura y escritura, preservando los tipos de datos.\n",
                "\n",
                "Además, definimos un diccionario de mapeo para traducir las columnas de su nombre original en el dataset procesado (muchas veces en español o códigos) a nombres técnicos en inglés estandarizados (ej. `Presion_Sistolica` -> `SystolicBP`). Esto facilita la interoperabilidad con librerías de ML y la consistencia en el proyecto.",
            ]
            break

    # 3. Update "Validación de Outliers..."
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown" and cell["source"][0].startswith(
            "## 2. Validación"
        ):
            cell["source"] = [
                "## 2. Validación Estadística y Detección de Outliers\n",
                "\n",
                'Antes de modelar, es crítico entender la "forma" de nuestros datos. Utilizamos `.describe()` para obtener un resumen estadístico de las variables numéricas:\n',
                "- **Count**: ¿Tenemos datos faltantes?\n",
                "- **Mean/Std**: ¿Cuál es el valor típico y qué tanto varían los datos?\n",
                "- **Min/Max**: ¿Existen valores fisiológicamente imposibles? (Ej. BMI < 10 o Glucosa = 0).\n",
                "\n",
                "Este paso nos permite identificar errores de calidad de datos o necesidad de limpieza adicional.",
            ]
            break

    # 4. Update "Análisis de Correlación..."
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown" and cell["source"][0].startswith(
            "## 3. Análisis"
        ):
            cell["source"] = [
                "## 3. Análisis de Correlación (Pearson)\n",
                "\n",
                "Buscamos **multicolinealidad** (variables que explican lo mismo) y relaciones fuertes con el target.\n",
                "- Usamos el coeficiente de correlación de **Pearson**.\n",
                "- Un valor cercano a **1** indica correlación positiva fuerte.\n",
                "- Un valor cercano a **-1** indica correlación negativa fuerte.\n",
                "- Un valor cercano a **0** indica ausencia de relación lineal.\n",
                "\n",
                "**Nota**: Variables muy correlacionadas (ej. `SystolicBP` y `DiastolicBP`) podrían introducir redundancia en ciertos modelos lineales, aunque algoritmos de árboles como XGBoost suelen manejarlas bien.",
            ]
            break

    # 5. Update "Reporte Sweetviz..."
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown" and cell["source"][0].startswith(
            "## 4. Reporte"
        ):
            cell["source"] = [
                "## 4. Reporte Automatizado con Sweetviz\n",
                "\n",
                "Generamos un reporte HTML interactivo utilizando la librería `Sweetviz`.\n",
                "- **Target**: `HeartDisease` (0 = Sano, 1 = Enfermo).\n",
                "- **Objetivo**: Comparar las distribuciones de cada feature para ambas clases.\n",
                "- **Interpretación**: Si las curvas de distribución para clase 0 y 1 se separan significativamente en una variable, esa variable es un buen predictor.\n",
                "\n",
                "El reporte se guardará como `NHANES_Clinical_Analysis.html` y puede abrirse en cualquier navegador web.",
            ]
            break

    with open(filepath, "w") as f:
        json.dump(nb, f, indent=1)


if __name__ == "__main__":
    update_notebook("notebooks/01_EDA_Clinical.ipynb")
