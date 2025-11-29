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
            'TCHOL': 'P_TCHOL.XPT', 'TRIGLY': 'P_TRIGLY.XPT', 'GHB': 'P_GHB.XPT',
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
            'TCHOL': 'TCHOL_I.XPT', 'TRIGLY': 'TRIGLY_I.XPT', 'GHB': 'GHB_I.XPT',
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
            'TCHOL': 'TCHOL_H.XPT', 'TRIGLY': 'TRIGLY_H.XPT', 'GHB': 'GHB_H.XPT',
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
            'TCHOL': 'TCHOL_G.XPT', 'TRIGLY': 'TRIGLY_G.XPT', 'GHB': 'GHB_G.XPT',
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
        df_ghb = load('GHB', ['LBXGH'])
        
        bio_cols = ['LBXSGL', 'LBXSCR', 'LBXSUA', 'LBXSATSI', 'LBXSAL', 'LBXSKSI', 'LBXSNASI', 'LBXSGTSI', 'LBXSASSI']
        df_bio = load('BIOPRO', bio_cols)

        # 5. Estilo de Vida
        df_smq = load('SMQ', ['SMQ020'])
        df_hiq = load('HIQ', ['HIQ011'])
        
        df_paq_raw = load('PAQ')
        if 'PAQ650' in df_paq_raw.columns:
            df_paq = df_paq_raw[['SEQN', 'PAQ650']].rename(columns={'PAQ650': 'Actividad_Fisica'})
        elif 'PAQ505' in df_paq_raw.columns: 
            df_paq = df_paq_raw[['SEQN', 'PAQ505']].rename(columns={'PAQ505': 'Actividad_Fisica'})
        else:
            df_paq = pd.DataFrame(columns=['SEQN', 'Actividad_Fisica'])

        # --- UNIÓN ---
        df_cycle = df_demo.merge(df_mcq, on='SEQN', how='inner') \
                          .merge(df_bp, on='SEQN', how='inner') \
                          .merge(df_bmx, on='SEQN', how='inner') \
                          .merge(df_chol, on='SEQN', how='inner') \
                          .merge(df_trig, on='SEQN', how='inner') \
                          .merge(df_ghb, on='SEQN', how='inner') \
                          .merge(df_bio, on='SEQN', how='inner') \
                          .merge(df_smq, on='SEQN', how='left') \
                          .merge(df_paq, on='SEQN', how='left') \
                          .merge(df_hiq, on='SEQN', how='left')
        
        print(f"   -> Filas recolectadas: {len(df_cycle)}")
        dfs_list.append(df_cycle)

    except Exception as e:
        print(f"   ❌ Error fatal en ciclo {y}: {e}")

# --- APILADO FINAL ---
print("\n🏗️ Concatenando todos los ciclos (2011-2020)...")
if dfs_list:
    df_final = pd.concat(dfs_list, ignore_index=True)

    # Renombrado Final
    cols_esp = {
        'RIAGENDR': 'Sexo', 'RIDAGEYR': 'Edad', 'RIDRETH1': 'Raza', 
        'DMDEDUC2': 'Educacion', 'INDFMPIR': 'Ingresos_Ratio',
        'BMXBMI': 'BMI', 'BMXWAIST': 'Cintura', 'BMXHT': 'Altura',
        'LBXTC': 'Colesterol_Total', 'LBXTR': 'Trigliceridos', 'LBDLDL': 'LDL', 'LBXGH': 'HbA1c',
        'LBXSGL': 'Glucosa', 'LBXSCR': 'Creatinina', 'LBXSUA': 'Acido_Urico',
        'LBXSATSI': 'Enzima_ALT', 'LBXSASSI': 'Enzima_AST', 'LBXSGTSI': 'Enzima_GGT',
        'LBXSAL': 'Albumina', 'LBXSKSI': 'Potasio', 'LBXSNASI': 'Sodio',
        'SMQ020': 'Fumador', 'HIQ011': 'Seguro_Medico'
    }
    df_final = df_final.rename(columns=cols_esp)

    # Limpieza Binaria y Nulos
    df_final['TARGET'] = df_final['TARGET'].apply(lambda x: 1 if x == 1 else 0)
    
    for col in ['Fumador', 'Seguro_Medico', 'Actividad_Fisica']:
        if col in df_final.columns:
            df_final[col] = df_final[col].apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))

    df_final = df_final.dropna()

    print(f"\n✅ DATASET 'ULTIMATE' LISTO (PARQUET).")
    print(f"Dimensiones Finales: {df_final.shape}")
    print(f"Variables Recuperadas: {list(df_final.columns)}")
    print(f"Total Positivos (Infartos): {df_final['TARGET'].sum()}")
    
    # --- CAMBIO AQUÍ: GUARDAR EN PARQUET ---
    df_final.to_parquet('NHANES_ULTIMATE_CLEAN.parquet', index=False)
else:
    print("❌ Falló la recolección.")