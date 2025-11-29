import sys
import os
import json
from typing import Optional, Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import shap
from streamlit_shap import st_shap
from src.adapters import PyCaretAdapter, UserInputAdapter

# Page Config
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# Load Config
CONFIG_PATH = "models/model_config.json"
MODEL_PATH = "models/final_pipeline_v1.pkl"

@st.cache_resource
def load_config() -> Dict[str, Any]:
    """
    Loads the model configuration from a JSON file.

    Returns:
        Dict[str, Any]: A dictionary containing configuration parameters (e.g., threshold).
                        Returns a default dictionary if the file is missing.
    """
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {"threshold": 0.5}

config = load_config()
default_threshold = config.get("threshold", 0.5)

# Load Model
@st.cache_resource
def load_model_pipeline() -> Optional[PyCaretAdapter]:
    """
    Loads the PyCaret model pipeline and wraps it in an adapter.

    Returns:
        Optional[PyCaretAdapter]: The adapted model ready for prediction, or None if loading fails.
    """
    if not os.path.exists(MODEL_PATH):
        # Fallback for development if specific model missing, mostly for testing UI logic
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

# Sidebar - Issue 41
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png")
    st.warning("⚠️ **Aviso Importante**\nEsta herramienta es un prototipo de apoyo al diagnóstico. No sustituye la opinión de un profesional médico.")
    if model:
        st.info(f"Modelo cargado: {type(model.model).__name__}")
    st.markdown("---")
    st.write("Desarrollado para Sprint 6")

# Title
st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter patient data below to estimate the risk of heart disease.")

# Issue 41.5: User Guide
with st.expander("📘 ¿Cómo interpretar los resultados?"):
    st.markdown("""
    - **Probabilidad de Riesgo:** Porcentaje calculado por el modelo.
    - **Umbral (Threshold):** Nivel a partir del cual se activa la alerta roja. Si priorizas encontrar a todos los enfermos (Sensibilidad), baja este valor.
    - **BMI:** Índice de Masa Corporal. Valor normal entre 18.5 y 24.9.
    """)

# Input Form
with st.form("patient_data_form"):
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, help="kg/m²")
        sex = st.selectbox("Sex", options=["Female", "Male"])

    with col2:
        smoker = st.selectbox("Smoker?", options=["No", "Yes"])
        diabetes = st.selectbox("Diabetes?", options=["No", "Yes"])
        phys_activity = st.selectbox("Physical Activity?", options=["No", "Yes"])

    # Threshold slider
    st.markdown("---")
    st.subheader("Decision Parameters")

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(default_threshold),
        step=0.01,
        help=f"Optimized Threshold from training: {default_threshold:.4f}"
    )

    # Submit
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    if model:
        try:
            # Prepare user input dict
            user_input = {
                'Age': age,
                'BMI': bmi,
                'Sex': sex,
                'Smoker': smoker,
                'Diabetes': diabetes,
                'PhysicalActivity': phys_activity
            }

            # Transform input
            adapter = UserInputAdapter()
            input_df = adapter.transform(user_input)

            # Ensure all columns required by the model are present
            if hasattr(model.model, 'feature_names_in_'):
                expected_cols = model.model.feature_names_in_
                input_df = input_df.reindex(columns=expected_cols, fill_value=0)

            # Predict
            prob = model.predict_proba(input_df)[0]
            prediction_label = "High Risk 🔴" if prob >= threshold else "Low Risk 🟢"

            st.toast("Predicción completada", icon="🚀")

            st.divider()
            st.subheader("Results")

            # Visual improvement with containers (Issue 41)
            result_container = st.container()
            with result_container:
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Risk Probability", f"{prob:.2%}")

                with col_res2:
                    if prob >= threshold:
                        st.error(f"**{prediction_label}**")
                    else:
                        st.success(f"**{prediction_label}**")

            st.info(f"Prediction made with threshold {threshold}. Optimized threshold was {default_threshold:.4f}.")

            # SHAP Integration (Issue 43)
            st.subheader("Explicación del Resultado")
            if st.checkbox("Ver por qué el modelo tomó esta decisión"):
                with st.spinner('Calculando importancia de variables...'):
                    try:
                        # 1. Extract underlying estimator and transformed data
                        pipeline = model.model

                        # Logic to handle PyCaret Pipeline to get estimator and transformed X
                        if hasattr(pipeline, 'steps'):
                            # Standard sklearn/imblearn pipeline
                            estimator = pipeline.steps[-1][1]

                            # Transform data through previous steps
                            X_transformed = input_df.copy()
                            for name, step in pipeline.steps[:-1]:
                                if hasattr(step, 'transform'):
                                    X_transformed = step.transform(X_transformed)
                                    # Handle numpy output from intermediate steps
                                    if isinstance(X_transformed, np.ndarray):
                                        # If we lose column names, we might have issues with waterfall plot if it expects names
                                        # But often steps preserve shape. If columns lost, we might need to rely on estimator.feature_names_in_
                                        pass

                            # If X_transformed is numpy, try to convert back to DF if estimator has feature names
                            if isinstance(X_transformed, np.ndarray) and hasattr(estimator, 'feature_names_in_'):
                                X_transformed = pd.DataFrame(X_transformed, columns=estimator.feature_names_in_)

                        else:
                            # If not a pipeline (just the model)
                            estimator = pipeline
                            X_transformed = input_df

                        # 2. Create Explainer
                        explainer = shap.TreeExplainer(estimator)

                        # 3. Calculate SHAP values for this instance
                        shap_values = explainer.shap_values(X_transformed)

                        # Handle SHAP output format (list for binary classification or array)
                        if isinstance(shap_values, list):
                            # Binary classification usually returns [class0, class1]
                            # We want class 1 (Risk)
                            sv = shap_values[1][0]
                            base_value = explainer.expected_value[1]
                        else:
                            sv = shap_values[0]
                            base_value = explainer.expected_value

                        # 4. Visualize with Waterfall
                        # Ensure we pass the first row of data for the plot
                        if isinstance(X_transformed, pd.DataFrame):
                            data_row = X_transformed.iloc[0]
                        else:
                            data_row = X_transformed[0]

                        explanation = shap.Explanation(values=sv,
                                                     base_values=base_value,
                                                     data=data_row,
                                                     feature_names=getattr(X_transformed, "columns", None))

                        st_shap(shap.plots.waterfall(explanation))

                    except Exception as e:
                        st.error(f"Error calculating SHAP values: {e}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Model could not be loaded. Please check if the model file exists.")
