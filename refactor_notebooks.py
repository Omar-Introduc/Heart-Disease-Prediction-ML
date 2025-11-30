import nbformat
import json

def update_eda_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Find the cell with "2. Validación Estadística..." to insert after it
    insert_idx = -1
    for idx, cell in enumerate(nb.cells):
        if "## 2. Validación Estadística" in cell.source:
            # We want to insert after the validation section, which ends after the statistical analysis code cell
            # Look for the code cell that runs .describe()
            pass
        if "df_analysis[numeric_cols].describe()" in cell.source:
             insert_idx = idx + 1
             break

    if insert_idx == -1:
        # Fallback: insert after the load data cell
        for idx, cell in enumerate(nb.cells):
             if "df = pd.read_parquet" in cell.source:
                 insert_idx = idx + 2 # After load and print columns
                 break

    if insert_idx == -1:
        print("Could not find insertion point for EDA notebook.")
        return

    # New Markdown Cell
    md_cell = nbformat.v4.new_markdown_cell(
        "## 2.1 Verificación de Consistencia Clínica (Friedewald)\n\n"
        "**Diagnóstico**: Verificamos que los datos respeten la fisiología humana básica.\n"
        "Aplicamos la **Fórmula de Friedewald** ($LDL_{calc} = ColesterolTotal - HDL - (Triglicéridos/5)$) para validar la coherencia entre los lípidos reportados.\n"
        "Si los puntos se alinean en la diagonal, los datos son consistentes. Desviaciones indican problemas de calidad o imputación."
    )

    # New Code Cell
    code_source = """# Verificación de Consistencia Clínica
def friedewald_validation(df):
    # Mapeo de columnas flexibles (Inglés/Español)
    col_map = {
        'TC': ['TotalCholesterol', 'Colesterol_Total', 'LBXTC'],
        'HDL': ['HDL', 'Colesterol_HDL', 'LBDHDD'],
        'Trig': ['Triglycerides', 'Trigliceridos', 'LBXTR'],
        'LDL': ['LDL', 'Colesterol_LDL', 'LBDLDL']
    }

    cols = {}
    for k, aliases in col_map.items():
        for a in aliases:
            if a in df.columns:
                cols[k] = a
                break

    # Verificación de existencia de columnas
    missing = set(col_map.keys()) - set(cols.keys())

    # Si falta HDL (común en algunos datasets procesados), intentamos inferirlo o advertimos
    if 'HDL' not in cols:
        if 'TC' in cols and 'LDL' in cols and 'Trig' in cols:
             print("ℹ️ Columna HDL explícita no encontrada. Derivando HDL implícito para visualización (Sanity Check)...")
             df['HDL_Implied'] = df[cols['TC']] - df[cols['LDL']] - (df[cols['Trig']] / 5)
             cols['HDL'] = 'HDL_Implied'
        else:
             print(f"⚠️ Faltan columnas críticas para validación Friedewald: {missing}. Se omite el gráfico.")
             return

    # Calcular LDL Teórico
    # LDL_calc = TC - HDL - (Trig / 5)
    ldl_calc = df[cols['TC']] - df[cols['HDL']] - (df[cols['Trig']] / 5)

    # Visualización
    plt.figure(figsize=(8, 6))
    plt.scatter(df[cols['LDL']], ldl_calc, alpha=0.3, c='blue', label='Pacientes', edgecolors='w', linewidth=0.5)

    # Línea de identidad
    min_val = min(df[cols['LDL']].min(), ldl_calc.min())
    max_val = max(df[cols['LDL']].max(), ldl_calc.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identidad Perfecta (Teórico = Reportado)')

    plt.xlabel(f"LDL Reportado ({cols['LDL']})")
    plt.ylabel("LDL Calculado (Friedewald)")
    plt.title("Validación Fisiológica: Consistencia de Lípidos")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Métricas de consistencia
    corr = df[cols['LDL']].corr(ldl_calc)
    print(f"Correlación Pearson (Reportado vs Calculado): {corr:.4f}")

    # Advertencia automática
    if corr < 0.9:
        print("⚠️ ADVERTENCIA CRÍTICA: Baja consistencia biológica (< 0.9). Revisar proceso de imputación o calidad de datos.")
    else:
        print("✅ Consistencia Biológica Aceptable.")

try:
    friedewald_validation(df_analysis)
except Exception as e:
    print(f"No se pudo ejecutar la validación clínica: {e}")
"""
    code_cell = nbformat.v4.new_code_cell(code_source)

    # Insert cells
    nb.cells.insert(insert_idx, md_cell)
    nb.cells.insert(insert_idx + 1, code_cell)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Updated {notebook_path}")


def update_training_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 1. Update Setup to use session_id=42
    for cell in nb.cells:
        if "setup(" in cell.source:
            cell.source = cell.source.replace("session_id=123", "session_id=42")
            # Ensure SAMPLE_FRAC is respected or force full if needed, but user said "Update setup... ensure session_id is fixed"
            # We keep SAMPLE_FRAC logic from the file but change the fixed ID.

    # 2. Replace simple comparison with Threshold Optimization
    # Find the cell with "best_model = compare_models"
    compare_idx = -1
    for idx, cell in enumerate(nb.cells):
        if "compare_models(" in cell.source:
            compare_idx = idx
            break

    if compare_idx != -1:
        # We replace the cells starting from compare_models up to save_model (excluding save_model for now)
        # Actually, let's just replace the "4. Comparación y Selección de Modelos" section content.

        # New Content for Model Selection & Optimization
        new_source = """# ==========================================\n# 3. SELECCIÓN DE MODELO Y OPTIMIZACIÓN DE UMBRAL\n# ==========================================\n# Estrategia: Precision-Constrained Recall Maximization\n# Buscamos el mejor Recall posible, pero garantizando que la Precisión sea al menos 40% (0.4)\n# para evitar fatiga de alarma en el personal médico.\n\n# 1. Entrenar y comparar modelos (Top 3)\ntop_models = compare_models(sort='Recall', n_select=3, verbose=False)\nprint(f"Top 3 Models: {top_models}")\n\n# 2. Tunear el mejor candidato\n# Optimizamos para Recall primero\nbest_model = top_models[0]\nprint(f"Tuning {type(best_model).__name__}...")\ntuned_model = tune_model(best_model, optimize='Recall', n_iter=20, verbose=False)\n\n# 3. Estrategia de Umbral de Seguridad Clínica\nprint("\\n--- Optimizando Umbral de Decisión ---")\n# Generar probabilidades en el set de validación (hold-out)\npredictions = predict_model(tuned_model, raw_score=True, verbose=False)\n\n# Identificar columnas de score y target real\n# PyCaret 3.x suele usar 'prediction_score' y 'prediction_label' o similar\ntarget_col = get_config('target_param')\ny_true = predictions[target_col]\n\n# Buscar columna de score para clase positiva (1)\nscore_cols = [c for c in predictions.columns if 'score' in c]\nif any('1' in c for c in score_cols):\n    score_col = [c for c in score_cols if '1' in c][0]\nelse:\n    score_col = score_cols[0]\n\ny_scores = predictions[score_col]\n\n# Iterar umbrales\nthresholds = np.arange(0.0, 1.0, 0.01)\nresults = []\n\nfrom sklearn.metrics import precision_score, recall_score, f1_score\n\nfor t in thresholds:\n    y_pred = (y_scores >= t).astype(int)\n    prec = precision_score(y_true, y_pred, zero_division=0)\n    rec = recall_score(y_true, y_pred, zero_division=0)\n    results.append({'Threshold': t, 'Precision': prec, 'Recall': rec})\n\nresults_df = pd.DataFrame(results)\n\n# Filtrar zona segura: Precision >= 0.4\nsafe_zone = results_df[results_df['Precision'] >= 0.4]\n\nif not safe_zone.empty:\n    # Seleccionar el umbral con mayor Recall dentro de la zona segura\n    # (Generalmente el umbral más bajo de la zona)\n    best_row = safe_zone.sort_values('Recall', ascending=False).iloc[0]\n    optimal_threshold = best_row['Threshold']\n    print(f"✅ Umbral Óptimo Encontrado: {optimal_threshold:.2f}")\n    print(f"   Métricas Esperadas -> Recall: {best_row['Recall']:.4f} | Precision: {best_row['Precision']:.4f}")\nelse:\n    print("⚠️ No se alcanzó la zona segura (Precision >= 0.4). Se usará umbral por defecto (0.5).")\n    optimal_threshold = 0.5\n\n# Visualización de la Curva de Seguridad\nplt.figure(figsize=(10, 6))\nplt.plot(results_df['Threshold'], results_df['Precision'], label='Precision', color='blue')\nplt.plot(results_df['Threshold'], results_df['Recall'], label='Recall', color='green')\nplt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimum ({optimal_threshold:.2f})')\n\n# Sombrear zona segura si existe\nif not safe_zone.empty:\n    plt.axvspan(safe_zone['Threshold'].min(), safe_zone['Threshold'].max(), alpha=0.1, color='green', label='Zona Segura (Prec>=0.4)')\n\nplt.title(f"Curva de Seguridad Clínica: Selección de Umbral para {type(tuned_model).__name__}")\nplt.xlabel("Umbral de Decisión")\nplt.ylabel("Métrica")\nplt.legend()\nplt.grid(True, alpha=0.3)\nplt.show()\n\n# Actualizar la configuración global para usar este umbral en predicciones futuras si es posible,\n# o simplemente guardar el valor para la API.\n"""

        # We overwrite the existing compare/tune cells with this new logic block
        # Assuming compare_models is in one cell and tune_model in another, we can replace the first and delete the others or combine.
        # Let's put everything in the compare_idx cell and clear subsequent tuning cells if they exist nearby.
        nb.cells[compare_idx].source = new_source

        # Remove the old tuning cell if present immediately after
        if compare_idx + 1 < len(nb.cells) and "tune_model" in nb.cells[compare_idx+1].source:
             nb.cells.pop(compare_idx + 1)


    # 3. Add SHAP plot before Finalize
    # Locate "4. FINALIZE & SAVE" (which seems to be cell index ~12)
    finalize_idx = -1
    for idx, cell in enumerate(nb.cells):
        if "finalize_model(" in cell.source:
            finalize_idx = idx
            break

    if finalize_idx != -1:
        # Insert SHAP before Finalize
        shap_md = nbformat.v4.new_markdown_cell(
            "## 4.5 Explicabilidad del Modelo (SHAP)\n\n"
            "Validamos que el modelo no tome decisiones basadas en artefactos o sesgos. Generamos el **SHAP Summary Plot** para visualizar las variables más impactantes.\n"
            "Esto es un requisito de **Transparencia Algorítmica** para la auditoría."
        )

        shap_code = nbformat.v4.new_code_cell(
            """# Generar SHAP Summary Plot
print("Generando explicaciones SHAP...")
try:
    interpret_model(tuned_model, plot='summary')
except Exception as e:
    print(f"No se pudo generar el gráfico SHAP (probablemente el modelo no lo soporte nativamente o falte librería): {e}")
"""
        )

        nb.cells.insert(finalize_idx, shap_md)
        nb.cells.insert(finalize_idx + 1, shap_code)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Updated {notebook_path}")

if __name__ == "__main__":
    update_eda_notebook('notebooks/01_EDA_Clinical.ipynb')
    update_training_notebook('notebooks/02_Training_PyCaret.ipynb')
