import sys
import pandas as pd
from pycaret.classification import load_model

model_path = "models/best_pipeline" # Relative to project root
try:
    print(f"Loading model from {model_path}...")
    pipeline = load_model(model_path)
    print("Model loaded successfully.")
    
    # Helper from previous fix
    def get_base_estimator(estimator):
        if hasattr(estimator, 'steps'):
            return estimator.steps[-1][1]
        return estimator

    model_step = pipeline.steps[-1][1]
    final_model = get_base_estimator(model_step)
    
    print(f"Final model type: {type(final_model)}")
    print(f"Has feature_importances_: {hasattr(final_model, 'feature_importances_')}")
    print(f"Has coef_: {hasattr(final_model, 'coef_')}")
    
    # List interesting attributes
    print("Attributes:", [d for d in dir(final_model) if not d.startswith('_')])

except Exception as e:
    print(f"Error inspecting model: {e}")
