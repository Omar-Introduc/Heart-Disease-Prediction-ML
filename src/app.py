import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from src.adapters import PyCaretAdapter, transform_user_input


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
    # Default threshold 0.25 as a starting point for Recall optimization, or 0.5 default
    threshold = st.slider("Decision Threshold (Optimize for Recall)", 0.0, 1.0, 0.5, 0.01)

    # Submit
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    if model:
        try:
            # Get feature names from the underlying model
            feature_names = []
            if hasattr(model.model, 'feature_names_in_'):
                feature_names = list(model.model.feature_names_in_)
            else:
                 # Fallback if feature names are not available
                 # This might happen with some pipeline structures
                 # Try to inspect steps
                 try:
                     # Check if it's a pipeline and the last step has feature names
                     feature_names = list(model.model.steps[-1][1].feature_names_in_)
                 except:
                     st.warning("Could not retrieve feature names from model. Using empty feature list (Prediction might be inaccurate).")

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
            input_df = transform_user_input(user_input, feature_names)

            # Predict
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

            st.info(f"Prediction made with threshold {threshold}. Lowering threshold increases Sensitivity (Recall).")

            with st.expander("Technical Details"):
                st.write(f"Model used: {type(model.model).__name__}")
                st.write(f"Input Shape: {input_df.shape}")
                st.write("Processed Input Features (Top 5 Non-Zero):")
                # Show non-zero features for verification
                non_zero = input_df.loc[:, (input_df != 0).any(axis=0)]
                st.dataframe(non_zero)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            import traceback
            st.text(traceback.format_exc())
    else:
        st.error("Model could not be loaded.")
