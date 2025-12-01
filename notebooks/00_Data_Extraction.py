#!/usr/bin/env python
# coding: utf-8

# # 📥 Extracción y Procesamiento de Datos NHANES (2011-2020)
#
# Este notebook consolida la lógica de extracción de datos de múltiples ciclos de NHANES y aplica la limpieza y estandarización final (nombres en inglés) para preparar el dataset `process_data.parquet`.

# In[4]:


import pandas as pd
import numpy as np

print("🚀 Iniciando Ingeniería de Datos 'MAXIMUM CAPACITY' (2011-2020) -> PARQUET...")

# --- CONFIGURACIÓN DE 4 CICLOS (Añadido 2011-2012) ---
cycles_config = [
    {
        'year': '2017-2020',
        'base': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/',
        'files': {
            'DEMO': 'P_DEMO.XPT', 'MCQ': 'P_MCQ.XPT', 'BMX': 'P_BMX.XPT',
            'BP': 'P_BPXO.XPT', # Oscilométrica
            'TCHOL': 'P_TCHOL.XPT', 'TRIGLY': 'P_TRIGLY.XPT', 'HDL': 'P_HDL.XPT', 'GHB': 'P_GHB.XPT',
            'BIOPRO': 'P_BIOPRO.XPT', 'SMQ': 'P_SMQ.XPT', 'ALQ': 'P_ALQ.XPT',
            'PAQ': 'P_PAQ.XPT', 'HIQ': 'P_HIQ.XPT'
        },
        'vars_map': { 'BP_SYS': ['BPXOSY1', 'BPXOSY2', 'BPXOSY3'], 'TARGET': 'MCQ160E' }
    },
    {
        'year': '2015-2016',
        'base': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/',
        'files': {
            'DEMO': 'DEMO_I.XPT', 'MCQ': 'MCQ_I.XPT', 'BMX': 'BMX_I.XPT',
            'BP': 'BPX_I.XPT',
            'TCHOL': 'TCHOL_I.XPT', 'TRIGLY': 'TRIGLY_I.XPT', 'HDL': 'HDL_I.XPT', 'GHB': 'GHB_I.XPT',
            'BIOPRO': 'BIOPRO_I.XPT', 'SMQ': 'SMQ_I.XPT', 'ALQ': 'ALQ_I.XPT',
            'PAQ': 'PAQ_I.XPT', 'HIQ': 'HIQ_I.XPT'
        },
        'vars_map': { 'BP_SYS': ['BPXSY1', 'BPXSY2', 'BPXSY3'], 'TARGET': 'MCQ160E' }
    },
    {
        'year': '2013-2014',
        'base': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/',
        'files': {
            'DEMO': 'DEMO_H.XPT', 'MCQ': 'MCQ_H.XPT', 'BMX': 'BMX_H.XPT',
            'BP': 'BPX_H.XPT',
            'TCHOL': 'TCHOL_H.XPT', 'TRIGLY': 'TRIGLY_H.XPT', 'HDL': 'HDL_H.XPT', 'GHB': 'GHB_H.XPT',
            'BIOPRO': 'BIOPRO_H.XPT', 'SMQ': 'SMQ_H.XPT', 'ALQ': 'ALQ_H.XPT',
            'PAQ': 'PAQ_H.XPT', 'HIQ': 'HIQ_H.XPT'
        },
        'vars_map': { 'BP_SYS': ['BPXSY1', 'BPXSY2', 'BPXSY3'], 'TARGET': 'MCQ160E' }
    },
    {
        'year': '2011-2012',
        'base': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/',
        'files': {
            'DEMO': 'DEMO_G.XPT', 'MCQ': 'MCQ_G.XPT', 'BMX': 'BMX_G.XPT',
            'BP': 'BPX_G.XPT',
            'TCHOL': 'TCHOL_G.XPT', 'TRIGLY': 'TRIGLY_G.XPT', 'HDL': 'HDL_G.XPT', 'GHB': 'GHB_G.XPT',
            'BIOPRO': 'BIOPRO_G.XPT', 'SMQ': 'SMQ_G.XPT', 'ALQ': 'ALQ_G.XPT',
            'PAQ': 'PAQ_G.XPT', 'HIQ': 'HIQ_G.XPT'
        },
        'vars_map': { 'BP_SYS': ['BPXSY1', 'BPXSY2', 'BPXSY3'], 'TARGET': 'MCQ160F' }
    }
]

dfs_list = []

for c in cycles_config:
    y = c['year']
    print(f"\n📥 Procesando ciclo: {y}...")

    try:
        def load(key, cols=None):
            url = c['base'] + c['files'][key]
            try:
                df = pd.read_sas(url)
                if cols:
                    exist = [x for x in cols if x in df.columns]
                    if len(exist) < len(cols):
                        missing = set(cols) - set(exist)
                    return df[['SEQN'] + exist]
                return df
            except Exception as e:
                print(f"   ⚠️ No se pudo descargar {key}: {e}")
                return pd.DataFrame(columns=['SEQN'])

        # 1. Cargas Básicas
        df_demo = load('DEMO', ['RIAGENDR', 'RIDAGEYR', 'RIDRETH1', 'DMDEDUC2', 'INDFMPIR'])

        # 2. Target
        df_mcq_raw = load('MCQ')
        target_candidates = ['MCQ160E', 'MCQ160F', 'MCQ160B']
        found_target = [t for t in target_candidates if t in df_mcq_raw.columns]

        if found_target:
            df_mcq = df_mcq_raw[['SEQN', found_target[0]]].rename(columns={found_target[0]: 'TARGET'})
        else:
            print(f"   ❌ No se encontró la columna de Infarto en {y}")
            continue

        # 3. Presión (Armonizada)
        df_bp_raw = load('BP')
        sys_cols = [col for col in c['vars_map']['BP_SYS'] if col in df_bp_raw.columns]
        if sys_cols:
            df_bp_raw['Presion_Sistolica'] = df_bp_raw[sys_cols].mean(axis=1)
            df_bp = df_bp_raw[['SEQN', 'Presion_Sistolica']]
        else:
            df_bp = pd.DataFrame(columns=['SEQN', 'Presion_Sistolica'])

        # 4. Clínicos
        df_bmx = load('BMX', ['BMXBMI', 'BMXWAIST', 'BMXHT'])
        df_chol = load('TCHOL', ['LBXTC'])
        df_trig = load('TRIGLY', ['LBXTR', 'LBDLDL'])
        df_hdl = load('HDL', ['LBDHDD'])
        df_ghb = load('GHB', ['LBXGH'])

        bio_cols = ['LBXSGL', 'LBXSCR', 'LBXSUA', 'LBXSATSI', 'LBXSAL', 'LBXSKSI', 'LBXSNASI', 'LBXSGTSI', 'LBXSASSI']
        df_bio = load('BIOPRO', bio_cols)

        # 5. Estilo de Vida
        df_smq = load('SMQ', ['SMQ020'])
        df_hiq = load('HIQ', ['HIQ011'])

        # --- CORRECCIÓN: Agregamos carga de Alcohol (ALQ101) ---
        # ALQ101: "Had at least 12 alcohol drinks/1 yr?" (1=Yes, 2=No)
        df_alq = load('ALQ', ['ALQ101'])

        df_paq_raw = load('PAQ')
        if 'PAQ650' in df_paq_raw.columns:
            df_paq = df_paq_raw[['SEQN', 'PAQ650']].rename(columns={'PAQ650': 'Actividad_Fisica'})
        elif 'PAQ505' in df_paq_raw.columns:
            df_paq = df_paq_raw[['SEQN', 'PAQ505']].rename(columns={'PAQ505': 'Actividad_Fisica'})
        else:
            df_paq = pd.DataFrame(columns=['SEQN', 'Actividad_Fisica'])

        # --- UNIÓN ---
        # Usamos LEFT JOIN para no perder pacientes que les falte algún examen específico
        df_cycle = df_demo.merge(df_mcq, on='SEQN', how='inner') \
                          .merge(df_bp, on='SEQN', how='left') \
                          .merge(df_bmx, on='SEQN', how='left') \
                          .merge(df_chol, on='SEQN', how='left') \
                          .merge(df_trig, on='SEQN', how='left') \
                          .merge(df_hdl, on='SEQN', how='left') \
                          .merge(df_ghb, on='SEQN', how='left') \
                          .merge(df_bio, on='SEQN', how='left') \
                          .merge(df_smq, on='SEQN', how='left') \
                          .merge(df_paq, on='SEQN', how='left') \
                          .merge(df_hiq, on='SEQN', how='left') \
                          .merge(df_alq, on='SEQN', how='left')

        print(f"   -> Filas recolectadas: {len(df_cycle)}")
        dfs_list.append(df_cycle)

    except Exception as e:
        print(f"   ❌ Error fatal en ciclo {y}: {e}")

# --- APILADO FINAL ---
print("\n🏗️ Concatenando todos los ciclos (2011-2020)...")
if dfs_list:
    df_final = pd.concat(dfs_list, ignore_index=True)

    # --- CORRECCIÓN DEFINITIVA: Nombres en Inglés directos ---
    # Mapeamos de códigos NHANES -> Nombres del model_config.json
    cols_eng = {
        # Demográficos
        'RIAGENDR': 'Sex',
        'RIDAGEYR': 'Age',
        'RIDRETH1': 'Race',
        'DMDEDUC2': 'Education',
        'INDFMPIR': 'IncomeRatio',

        # Físicos
        'BMXBMI': 'BMI',
        'BMXWAIST': 'WaistCircumference',
        'BMXHT': 'Height',

        # Laboratorio
        'LBXTC': 'TotalCholesterol',
        'LBXTR': 'Triglycerides',
        'LBDHDD': 'HDL',
        'LBDLDL': 'LDL',
        'LBXGH': 'HbA1c',
        'LBXSGL': 'Glucose',
        'LBXSCR': 'Creatinine',
        'LBXSUA': 'UricAcid',
        'LBXSATSI': 'ALT_Enzyme',
        'LBXSASSI': 'AST_Enzyme',
        'LBXSGTSI': 'GGT_Enzyme',
        'LBXSAL': 'Albumin',
        'LBXSKSI': 'Potassium',
        'LBXSNASI': 'Sodium',

        # Estilo de Vida
        'SMQ020': 'Smoking',
        'HIQ011': 'HealthInsurance',
        'ALQ101': 'Alcohol', # Asegúrate de haber agregado la carga de ALQ arriba

        # Target
        'TARGET': 'HeartDisease',
        'Presion_Sistolica': 'SystolicBP',
        'Actividad_Fisica': 'PhysicalActivity'
    }

    df_final = df_final.rename(columns=cols_eng)

    # Limpieza Binaria y Nulos
    # Nota: Ahora usamos los nombres en Inglés
    df_final['HeartDisease'] = df_final['HeartDisease'].apply(lambda x: 1 if x == 1 else 0)

    binary_cols = ['Smoking', 'HealthInsurance', 'PhysicalActivity', 'Alcohol']
    for col in binary_cols:
        if col in df_final.columns:
            # 1=Yes, 2=No -> 1=1, 0=2
            df_final[col] = df_final[col].apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))

    # df_final = df_final.dropna() # Comentado para no perder datos. PyCaret imputará los nulos.

    # Solo eliminamos si falta el TARGET (esencial para entrenamiento)
    df_final = df_final.dropna(subset=['HeartDisease'])

    print(f"\n✅ DATASET 'ULTIMATE' LISTO (ENGLISH VERSION).")
    print(f"Dimensiones Finales: {df_final.shape}")
    print(f"Variables: {list(df_final.columns)}")

    # --- INYECCIÓN DE LÓGICA DE IMPUTACIÓN CLÍNICA ---
    import sys
    import os

    # Asegurar que 'src' es visible.
    project_root = os.path.abspath('..')
    if project_root not in sys.path:
        sys.path.append(project_root)

    from src.data_ingestion import impute_clinical_data

    print("\n🩺 Aplicando Imputación Clínica Avanzada (Friedewald + IterativeImputer)...")
    df_final = impute_clinical_data(df_final)
    print("✅ Imputación completada.")

    # Guardamos en la carpeta intermediate
    output_path = '../data/02_intermediate/process_data.parquet'
    df_final.to_parquet(output_path, index=False)
    print(f"\n💾 Guardado exitosamente en: {output_path}")
else:
    print("❌ Falló la recolección.")
