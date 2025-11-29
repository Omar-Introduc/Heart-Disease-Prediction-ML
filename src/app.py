import sys
import os
import json
from typing import Optional, Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np

try:
    import shap
    from streamlit_shap import st_shap
    SHAP_AVAILABLE = True
except (ImportError, OSError):
    SHAP_AVAILABLE = False

from src.adapters import PyCaretAdapter, UserInputAdapter

# Page Config
st.set_page_config(
    page_title="Heart Disease Risk Prediction (NHANES)",
    page_icon="❤️",
    layout="wide"
)

# Load Config
CONFIG_PATH = "models/model_config.json"
MODEL_PATH = "models/final_pipeline_v1.pkl"

@st.cache_resource
def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {"threshold": 0.5}

config = load_config()
default_threshold = config.get("threshold", 0.5)

# Load Model
@st.cache_resource
def load_model_pipeline() -> Optional[PyCaretAdapter]:
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        from pycaret.classification import load_model
        path_without_ext = os.path.splitext(MODEL_PATH)[0]
        pipeline = load_model(path_without_ext)
        return PyCaretAdapter(pipeline)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model_pipeline()

if model:
    st.toast("Modelo cargado exitosamente", icon="✅")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png")
    st.warning("⚠️ **Aviso Importante**\nEsta herramienta es un prototipo basado en datos clínicos (NHANES). No sustituye la opinión médica.")
    if model:
        st.info(f"Modelo cargado: {type(model.model).__name__}")
    st.markdown("---")
    st.write("Refactorizado para NHANES Schema")

# Title
st.title("❤️ Heart Disease Risk Prediction")
st.markdown("Enter clinical patient data below to estimate risk.")

# Input Form
with st.form("patient_data_form"):

    # 1. Datos Personales
    st.subheader("👤 Datos Personales")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=45)
    with col2:
        sex_radio = st.radio("Sex", options=["Female", "Male"], horizontal=True)
        sex = 1 if sex_radio == "Male" else 0
    with col3:
        height = st.number_input("Height (cm)", min_value=130.0, max_value=220.0, value=170.0)
    with col4:
        waist = st.number_input("Waist Circumference (cm)", min_value=50.0, max_value=180.0, value=90.0)

    # 2. Signos Vitales
    st.subheader("🫀 Signos Vitales")
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        bmi = st.number_input("BMI", min_value=12.0, max_value=60.0, value=25.0, format="%.1f")
    with col_v2:
        sys_bp = st.slider("Systolic BP (mmHg)", 80.0, 220.0, 120.0)
    with col_v3:
        dia_bp = st.slider("Diastolic BP (mmHg) [Opcional]", 40.0, 120.0, 80.0)

    # 3. Perfil Bioquímico
    st.subheader("🧪 Perfil Bioquímico")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        chol = st.number_input("Total Cholesterol (mg/dL)", 100.0, 400.0, 200.0)
        ldl = st.number_input("LDL (mg/dL)", 30.0, 300.0, 100.0)
    with col_b2:
        trig = st.number_input("Triglycerides (mg/dL)", 30.0, 600.0, 150.0)
        hba1c = st.number_input("HbA1c (%)", 4.0, 15.0, 5.7, step=0.1)
    with col_b3:
        glucose = st.number_input("Glucose (mg/dL)", 50.0, 300.0, 90.0)
        uric = st.number_input("Uric Acid (mg/dL)", 2.0, 12.0, 5.0, step=0.1)
    with col_b4:
        creat = st.number_input("Creatinine (mg/dL)", 0.4, 5.0, 0.9, step=0.1)

    # 4. Estilo de Vida y Antecedentes
    st.subheader("🏃 Estilo de Vida")
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    with col_l1:
        smoking = st.checkbox("Fuma / Fumó >100 cigarrillos?", value=False)
    with col_l2:
        alcohol = st.checkbox("Bebe alcohol frecuentemente?", value=False)
    with col_l3:
        activity = st.checkbox("Actividad física vigorosa?", value=False)
    with col_l4:
        insurance = st.checkbox("Tiene seguro médico?", value=True)

    # Convert checkboxes to int
    smoking_int = 1 if smoking else 0
    alcohol_int = 1 if alcohol else 0
    activity_int = 1 if activity else 0
    insurance_int = 1 if insurance else 0

    st.markdown("---")
    threshold = st.slider(
        "Decision Threshold (Sensitivity adjustment)",
        min_value=0.0,
        max_value=1.0,
        value=float(default_threshold),
        step=0.01
    )

    submitted = st.form_submit_button("Predict Clinical Risk")

if submitted:
    if model:
        try:
            # Prepare dictionary
            user_input = {
                'Age': age,
                'Sex': sex,
                'Height': height,
                'BMI': bmi,
                'SystolicBP': sys_bp,
                'DiastolicBP': dia_bp,
                'WaistCircumference': waist,
                'TotalCholesterol': chol,
                'LDL': ldl,
                'Triglycerides': trig,
                'HbA1c': hba1c,
                'Glucose': glucose,
                'UricAcid': uric,
                'Creatinine': creat,
                'Smoking': smoking_int,
                'Alcohol': alcohol_int,
                'PhysicalActivity': activity_int,
                'HealthInsurance': insurance_int
            }

            # Adapter
            adapter = UserInputAdapter()
            input_df = adapter.transform(user_input)

            # Ensure columns match model expectation (handling unexpected columns if any)
            if hasattr(model.model, 'feature_names_in_'):
                expected_cols = model.model.feature_names_in_
                # Reindex allows filling missing columns with 0 if needed, though Adapter should have covered it
                input_df = input_df.reindex(columns=expected_cols, fill_value=0)

            # Predict
            prob = model.predict_proba(input_df)[0]
            prediction_label = "High Risk 🔴" if prob >= threshold else "Low Risk 🟢"

            st.divider()
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Risk Probability", f"{prob:.2%}")
            with col_res2:
                if prob >= threshold:
                    st.error(f"**{prediction_label}**")
                else:
                    st.success(f"**{prediction_label}**")

            # SHAP
            if SHAP_AVAILABLE:
                if st.checkbox("Show Explanation (SHAP)"):
                    with st.spinner('Calculating explanation...'):
                        try:
                            # Simple SHAP implementation for the first row
                            pipeline = model.model
                            if hasattr(pipeline, 'steps'):
                                estimator = pipeline.steps[-1][1]
                            else:
                                estimator = pipeline

                            # Note: SHAP with pipelines can be tricky.
                            # Ideally use explainer on the estimator and transformed data.
                            # For simplicity, we skip full transformation logic in this snippet
                            # and assume TreeExplainer works if compatible.
                            explainer = shap.TreeExplainer(estimator)
                            shap_values = explainer.shap_values(input_df)

                            if isinstance(shap_values, list):
                                sv = shap_values[1][0]
                                base_value = explainer.expected_value[1]
                            else:
                                sv = shap_values[0]
                                base_value = explainer.expected_value

                            explanation = shap.Explanation(values=sv, base_values=base_value, data=input_df.iloc[0], feature_names=input_df.columns)
                            st_shap(shap.plots.waterfall(explanation))
                        except Exception as e:
                            st.warning(f"Could not generate explanation: {e}")
            else:
                if st.checkbox("Show Explanation (SHAP)", disabled=True, help="Feature unavailable due to missing dependencies (shap/numba)."):
                    pass

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Model not loaded.")
