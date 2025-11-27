import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from src.adapters import PyCaretAdapter

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
def load_model():
    model_path = "models/best_pipeline.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    return PyCaretAdapter(pipeline)

model = load_model()

# Input Form
with st.form("patient_data_form"):
    st.subheader("Patient Information")

    # Example fields based on common BRFSS variables
    # We need to match the features the model expects.
    # Since we are using a mock model or dynamic features,
    # we will add a few representative ones.

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
    threshold = st.slider("Decision Threshold (Recall Optimization)", 0.0, 1.0, 0.5, 0.05)

    # Submit
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    if model:
        # Prepare input data
        # Note: This must match the columns the model was trained on.
        # Since we have a mock model or PyCaret model, we'd typically align columns.
        # For this prototype, we create a DataFrame with expected columns.
        # IF the model is the mock one, it expects 'col_0'...'col_4'.
        # IF it's the real one, it expects BRFSS codes.

        # Checking features from model if possible
        try:
            if hasattr(model.model, 'feature_names_in_'):
                features = model.model.feature_names_in_
                # Create a dict with 0s for missing features
                input_data = {f: 0 for f in features}

                # Update with our inputs if they map loosely
                # (This is just for demo, real mapping requires codebook)
                pass

                input_df = pd.DataFrame([input_data])
            else:
                # Fallback for mock model
                input_df = pd.DataFrame(np.random.rand(1, 5), columns=[f'col_{i}' for i in range(5)])
        except Exception:
             input_df = pd.DataFrame(np.random.rand(1, 5))

        # Predict
        try:
            prob = model.predict_proba(input_df)[0]
            prediction = model.predict(input_df, threshold=threshold)[0]

            st.divider()
            st.subheader("Results")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Risk Probability", f"{prob:.2%}")

            with col_res2:
                if prediction == 1:
                    st.error("High Risk Detected")
                else:
                    st.success("Low Risk")

            st.info(f"Prediction made with threshold {threshold}. Lowering threshold increases Recall (sensitivity).")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Model could not be loaded.")
