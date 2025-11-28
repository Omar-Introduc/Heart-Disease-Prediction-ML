import sys
import os
import json
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
from src.adapters import PyCaretAdapter, UserInputAdapter

# Page Config
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# Title
st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter patient data below to estimate the risk of heart disease.")

# Load Config
CONFIG_PATH = "models/model_config.json"
MODEL_PATH = "models/final_pipeline_v1.pkl"

@st.cache_resource
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {"threshold": 0.5}

config = load_config()
default_threshold = config.get("threshold", 0.5)

# Load Model
@st.cache_resource
def load_model_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        return None

    try:
        # PyCaret saves as .pkl, but sometimes with .pkl suffix in filename or not
        # If load_model from pycaret was used, it adds .pkl.
        # But we use pickle directly here if we want or pycaret's load_model function.
        # However, to be safe and standard with the training script:
        from pycaret.classification import load_model
        # load_model appends .pkl automatically if not present in string, but expects path without extension usually
        # The training script saved as 'models/final_pipeline_v1' (no extension in string passed to save_model)

        path_without_ext = os.path.splitext(MODEL_PATH)[0]
        pipeline = load_model(path_without_ext)
        return PyCaretAdapter(pipeline)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model_pipeline()

# Input Form
with st.form("patient_data_form"):
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        sex = st.selectbox("Sex", options=["Female", "Male"]) # Female default

    with col2:
        smoker = st.selectbox("Smoker?", options=["No", "Yes"])
        diabetes = st.selectbox("Diabetes?", options=["No", "Yes"])
        phys_activity = st.selectbox("Physical Activity?", options=["No", "Yes"])

    # Threshold slider
    st.markdown("---")
    st.subheader("Decision Parameters")
    st.write("Adjust the threshold to balance between detecting more cases (High Recall) and reducing false alarms.")

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
            # We pad missing columns with 0
            if hasattr(model.model, 'feature_names_in_'):
                expected_cols = model.model.feature_names_in_
                # Reindex to match model columns, filling missing with 0
                input_df = input_df.reindex(columns=expected_cols, fill_value=0)

            # Predict
            prob = model.predict_proba(input_df)[0]
            prediction_label = "High Risk 🔴" if prob >= threshold else "Low Risk 🟢"

            st.divider()
            st.subheader("Results")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Risk Probability", f"{prob:.2%}")

            with col_res2:
                if prob >= threshold:
                    st.error(f"{prediction_label}")
                else:
                    st.success(f"{prediction_label}")

            st.info(f"Prediction made with threshold {threshold}. Optimized threshold was {default_threshold:.4f}.")

            with st.expander("Technical Details"):
                st.write(f"Model used: {type(model.model).__name__}")
                st.write("Processed Input Features:")
                st.dataframe(input_df)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            import traceback
            st.text(traceback.format_exc())
    else:
        st.error("Model could not be loaded.")
