from pycaret.classification import load_model
import sys
import os

model_path_no_ext = 'models/best_pipeline'

try:
    loaded_obj = load_model(model_path_no_ext)
    
    print(f"Loaded object type: {type(loaded_obj)}")
    print(f"Loaded object: {loaded_obj}")
    
    if hasattr(loaded_obj, 'predict'):
        print("Object has 'predict' method.")
    else:
        print("Object does NOT have 'predict' method.")

except Exception as e:
    print(f"Error loading model: {e}")
