import pandas as pd
import numpy as np
import os

# Lista de columnas identificadas en el notebook de EDA
columnas_enriquecidas = [
    'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', '_AGEG5YR', 'SEXVAR', '_BMI5',
    '_SMOKER3', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH', '_TOTINDA', 'SLEPTIM1',
    '_RFBING6', 'DIABETE4', 'HAVARTH4', 'ADDEPEV3', 'CHCKDNY2', '_EDUCAG',
    'MARITAL', '_INCOMG1', '_PRACE2'
]

# Generación de datos ficticios
# Se utilizan valores plausibles y aleatorios basados en el análisis previo del notebook
data = {
    'CVDINFR4': np.random.choice([1.0, 2.0], size=100),
    'CVDCRHD4': np.random.choice([1.0, 2.0, 7.0, 9.0], size=100, p=[0.1, 0.8, 0.05, 0.05]),
    'CVDSTRK3': np.random.choice([1.0, 2.0, 7.0, 9.0], size=100, p=[0.1, 0.8, 0.05, 0.05]),
    '_AGEG5YR': np.random.randint(1, 14, size=100).astype(float),
    'SEXVAR': np.random.choice([1.0, 2.0], size=100),
    '_BMI5': np.random.randint(1500, 4000, size=100).astype(float),
    '_SMOKER3': np.random.randint(1, 5, size=100).astype(float),
    'GENHLTH': np.random.randint(1, 6, size=100).astype(float),
    'PHYSHLTH': np.random.choice(list(range(1, 31)) + [77, 88, 99], size=100).astype(float),
    'MENTHLTH': np.random.choice(list(range(1, 31)) + [77, 88, 99], size=100).astype(float),
    '_TOTINDA': np.random.choice([1.0, 2.0, 9.0], size=100),
    'SLEPTIM1': np.random.randint(1, 24, size=100).astype(float),
    '_RFBING6': np.random.choice([1.0, 2.0, 9.0], size=100),
    'DIABETE4': np.random.randint(1, 5, size=100).astype(float),
    'HAVARTH4': np.random.choice([1.0, 2.0, 7.0, 9.0], size=100, p=[0.3, 0.6, 0.05, 0.05]),
    'ADDEPEV3': np.random.choice([1.0, 2.0, 7.0, 9.0], size=100, p=[0.2, 0.7, 0.05, 0.05]),
    'CHCKDNY2': np.random.choice([1.0, 2.0, 7.0, 9.0], size=100, p=[0.1, 0.8, 0.05, 0.05]),
    '_EDUCAG': np.random.randint(1, 7, size=100).astype(float),
    'MARITAL': np.random.randint(1, 7, size=100).astype(float),
    '_INCOMG1': np.random.randint(1, 7, size=100).astype(float),
    '_PRACE2': np.random.randint(1, 10, size=100).astype(float)
}

df_dummy = pd.DataFrame(data)

# Crear el directorio si no existe
output_path = 'data/01_raw'
os.makedirs(output_path, exist_ok=True)

# Guardar el DataFrame como un archivo CSV
ruta_dummy_csv = os.path.join(output_path, 'dummy_data.csv')
df_dummy.to_csv(ruta_dummy_csv, index=False)

print(f"Datos ficticios guardados en: '{ruta_dummy_csv}'")
