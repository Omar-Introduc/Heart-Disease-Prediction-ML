# Sistema de Machine Learning para la Predicción de Riesgo de Enfermed-ad Cardíaca

### Objetivo General
Desarrollar un sistema predictivo que permita estimar el riesgo de enfermedad cardíaca en pacientes a partir de datos clínicos y biomédicos, con el fin de apoyar la toma de decisiones médicas y contribuir a la detección temprana.

### Objetivos Específicos
1.  Recopilar y analizar datasets relevantes de enfermedades cardiovasculares.
2.  Preprocesar y normalizar los datos clínicos (limpieza, imputación, selección de características).
3.  Implementar y comparar diferentes algoritmos de clasificación (Regresión Logística, Random Forest, SVM, XGBoost).
4.  Optimizar y validar los modelos mediante validación cruzada y ajuste de hiperparámetros.
5.  Garantizar la interpretabilidad de los modelos aplicando herramientas como SHAP y LIME.
6.  Diseñar una interfaz de usuario y documentar el proceso.

## ⚙️ Instalación

Sigue estos pasos para configurar el entorno y preparar los datos para ejecutar el proyecto.

### 1. Clonar el Repositorio
```bash
git clone [TU_ENLACE_AL_REPOSITORIO_DE_GITHUB]
cd HEART-DISEASE-PREDICTION-ML
```

## ⚙️ Configuración del Entorno y Datos

### 1. Entorno
Para replicar el entorno de desarrollo, utiliza el archivo `environment.yml`:
```bash
conda env create -f environment.yml
conda activate xgb_env
```

### 3. Descargar el Dataset
Este proyecto utiliza el dataset **BRFSS 2022** de los CDC de EE. UU. Debido a su tamaño, el archivo de datos no está incluido en este repositorio.

Para obtener los datos:
1.  Ve a la página oficial de descarga: https://www.cdc.gov/brfss/annual_data/annual_2022.html
2.  Descarga el archivo **"2022 BRFSS Data (SAS Transport Format)"**.
3.  Descomprime el archivo ZIP y coloca el fichero `LLCP2022.XPT` resultante dentro de la carpeta `data/01_raw/`.