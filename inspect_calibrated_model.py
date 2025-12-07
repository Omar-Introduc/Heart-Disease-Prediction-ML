import sys
import pandas as pd
import numpy as np
from pycaret.classification import load_model

model_path = "models/best_pipeline"
try:
    print(f"Loading model from {model_path}...")
    pipeline = load_model(model_path)
    
    def get_base_estimator(estimator):
        if hasattr(estimator, 'steps'):
            return estimator.steps[-1][1]
        return estimator

    model_step = pipeline.steps[-1][1]
    final_model = get_base_estimator(model_step)
    
    print(f"Final model type: {type(final_model)}")
    
    if hasattr(final_model, 'calibrated_classifiers_'):
        print(f"Num calibrated classifiers: {len(final_model.calibrated_classifiers_)}")
        first_clf = final_model.calibrated_classifiers_[0]
        print(f"First calibrated type: {type(first_clf)}")
        print(f"Dir of first calibrated: {[d for d in dir(first_clf) if not d.startswith('_')]}")
        
        # Check for estimator/base_estimator
        if hasattr(first_clf, 'estimator'):
            print(f"Has .estimator: {type(first_clf.estimator)}")
            est = first_clf.estimator
            print(f"Estimator attributes: {[d for d in dir(est) if not d.startswith('_')]}")
            if hasattr(est, 'coef_'): print(f"Estimator has coef_: {est.coef_.shape}")
            if hasattr(est, 'feature_importances_'): print("Estimator has feature_importances_")
            
        if hasattr(first_clf, 'base_estimator'):
            print(f"Has .base_estimator: {type(first_clf.base_estimator)}")

    else:
        print("Model does not have calibrated_classifiers_")

except Exception as e:
    print(f"Error inspecting model: {e}")
