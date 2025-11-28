import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from src.adapters import get_adapter

# Page Config
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# Title
st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter patient data below to estimate the risk of heart disease.")

# Load Model
@st.cache_resource
def load_resources():
    model_path = "models/best_pipeline.pkl"
    feature_path = "models/feature_names.json"
    threshold_path = "models/threshold.json"

    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None, None, None

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    adapter = get_adapter(feature_path)

    threshold = 0.5
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            t_data = json.load(f)
            threshold = t_data.get('optimal_threshold', 0.5)

    return pipeline, adapter, threshold

model, adapter, default_threshold = load_resources()

# Input Form
with st.form("patient_data_form"):
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        sex = st.selectbox("Sex", options=["Male", "Female"])

    with col2:
        smoker = st.selectbox("Smoker?", options=["Yes", "No"])
        diabetes = st.selectbox("Diabetes?", options=["Yes", "No"])
        phys_activity = st.selectbox("Physical Activity?", options=["Yes", "No"])

    # Threshold slider
    threshold = st.slider(
        "Decision Threshold (Recall Optimization)",
        0.0, 1.0, float(default_threshold), 0.01,
        help="Scientific threshold optimized for F2-Score."
    )

    # Submit
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    if model and adapter:
        # Prepare input data using Mapper
        ui_inputs = {
            'Age': age,
            'BMI': bmi,
            'Sex': sex,
            'Smoker': smoker,
            'Diabetes': diabetes,
            'PhysicalActivity': phys_activity
        }

        input_df = adapter.map_ui_input_to_model(ui_inputs)

        # Predict
        try:
            # Check if model supports predict_proba
            # Access the estimator step if it's a pipeline
            # PyCaret pipeline usually has steps.
            # If RidgeClassifier, it might not have predict_proba.

            # Helper to get probability
            has_proba = False
            try:
                prob_array = model.predict_proba(input_df)
                # prob_array is (n_samples, n_classes)
                # We want prob of class 1
                prob = prob_array[0][1]
                has_proba = True
            except AttributeError:
                # Fallback for models without predict_proba (e.g. Ridge)
                # Use decision_function if available
                if hasattr(model, "decision_function"):
                     scores = model.decision_function(input_df)
                     # Sigmoid to approximate probability
                     prob = 1 / (1 + np.exp(-scores[0]))
                     has_proba = True # synthetic probability
                else:
                    # Just use predict
                    pred_label = model.predict(input_df)[0]
                    prob = 1.0 if pred_label == 1 else 0.0

            # Apply threshold
            prediction = 1 if prob >= threshold else 0

            st.divider()
            st.subheader("Results")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Risk Score", f"{prob:.2%}")

            with col_res2:
                if prediction == 1:
                    st.error("High Risk Detected")
                else:
                    st.success("Low Risk")

            st.info(f"Prediction made with threshold {threshold}. Optimal calculated was {default_threshold}.")

            # Show debug info
            with st.expander("Debug Input Vector"):
                st.write(input_df.T)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            import traceback
            st.text(traceback.format_exc())
    else:
        st.error("Model or Adapter could not be loaded.")
