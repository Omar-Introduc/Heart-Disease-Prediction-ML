import pandas as pd
import os
import json
import shutil
from pycaret.classification import load_model, predict_model
from sklearn.metrics import recall_score, confusion_matrix
def audit_fairness():
    print("=== Automated Fairness Audit (Gender Bias) ===")
    model_path = "models/best_pipeline"
    if os.path.exists("models/final_pipeline_v1.pkl"):
        model_path = "models/final_pipeline_v1"
    data_path = "data/02_intermediate/process_data.parquet"
    if not os.path.exists(f"{model_path}.pkl"):
        print(f"❌ Error: Model not found at {model_path}.pkl")
        return
    if not os.path.exists(data_path):
        print(f"❌ Error: Data not found at {data_path}")
        return
    print(f"Loading Model: {model_path}")
    pipeline = load_model(model_path)
    print(f"Loading Data: {data_path}")
    df = pd.read_parquet(data_path)
    rename_map = {
        "TARGET": "HeartDisease",
        "Edad": "Age",
        "Sexo": "Sex",
        "Raza": "Race",
        "Educacion": "Education",
        "Presion_Sistolica": "SystolicBP",
        "Cintura": "WaistCircumference",
        "Altura": "Height",
        "Colesterol_Total": "TotalCholesterol",
        "Trigliceridos": "Triglycerides",
        "Glucosa": "Glucose",
        "Creatinina": "Creatinine",
        "Acido_Urico": "UricAcid",
        "Fumador": "Smoking",
        "Actividad_Fisica": "PhysicalActivity",
        "Seguro_Medico": "HealthInsurance",
    }
    df = df.rename(columns=rename_map)
    target_col = "HeartDisease"
    if target_col not in df.columns:
        possible_targets = ["CVDINFR4", "Target", "Outcome", "HeartDisease", "TARGET"]
        for t in possible_targets:
            if t in df.columns:
                target_col = t
                break
    if target_col not in df.columns:
        print(f"❌ Error: Target column not found.")
        return
    print("Running predictions...")
    predictions = predict_model(pipeline, data=df, verbose=False)
    pred_col = "prediction_label"
    if pred_col not in predictions.columns:
         pred_col = [c for c in predictions.columns if 'label' in c][0]
    print("Calculating metrics per group...")
    df_male = predictions[predictions["Sex"] == 1]
    df_female = predictions[predictions["Sex"] == 0]
    y_true_m = df_male[target_col]
    y_pred_m = df_male[pred_col]
    y_true_f = df_female[target_col]
    y_pred_f = df_female[pred_col]
    recall_m = recall_score(y_true_m, y_pred_m, zero_division=0)
    recall_f = recall_score(y_true_f, y_pred_f, zero_division=0)
    fnr_m = 1 - recall_m
    fnr_f = 1 - recall_f
    eq_opp_ratio = recall_f / recall_m if recall_m > 0 else 0.0
    print(f"Men (Sex=1): Recall={recall_m:.4f}, FNR={fnr_m:.4f}")
    print(f"Women (Sex=0): Recall={recall_f:.4f}, FNR={fnr_f:.4f}")
    print(f"Equal Opportunity Ratio (Recall F/M): {eq_opp_ratio:.4f}")
    bias_ratio = fnr_f / fnr_m if fnr_m > 0 else 999.0 
    status = "✅ PASS"
    message = "Bias within acceptable limits."
    if fnr_f > 1.2 * fnr_m:
        status = "❌ FAIL"
        message = "Significant Bias Detected against Women (High FNR)."
    print(f"Bias Check: FNR_Female / FNR_Male = {bias_ratio:.2f}")
    print(f"Result: {status} - {message}")
    report = {
        "model_path": model_path,
        "metrics": {
            "men": {
                "recall": recall_m,
                "fnr": fnr_m,
                "count": len(df_male)
            },
            "women": {
                "recall": recall_f,
                "fnr": fnr_f,
                "count": len(df_female)
            }
        },
        "fairness_metrics": {
            "equal_opportunity_ratio": eq_opp_ratio,
            "fnr_ratio": bias_ratio
        },
        "status": status,
        "message": message
    }
    report_path = "docs/fairness_audit_report.json"
    if not os.path.exists("docs"):
        os.makedirs("docs")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Report saved to {report_path}")
if __name__ == "__main__":
    audit_fairness()
