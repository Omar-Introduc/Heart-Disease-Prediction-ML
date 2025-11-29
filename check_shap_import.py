
try:
    import shap
    print("shap imported successfully")
except Exception as e:
    print(f"Error importing shap: {e}")

try:
    from streamlit_shap import st_shap
    print("streamlit_shap imported successfully")
except Exception as e:
    print(f"Error importing streamlit_shap: {e}")
