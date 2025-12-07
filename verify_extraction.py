import sys
import pandas as pd
import numpy as np
from pycaret.classification import load_model
import traceback

# Setup explicit logging
log_buffer = []

def log(msg):
    print(msg)
    log_buffer.append(str(msg))

model_path = "models/best_pipeline"

def get_base_estimator(estimator):
    if hasattr(estimator, 'steps'):
        return estimator.steps[-1][1]
    return estimator

def get_model_importance_debug(model, depth=0):
    indent = "  " * depth
    log(f"{indent}Inspecting model type: {type(model)}")
    
    try:
        # Case 1: Tree-based (Fitted)
        if hasattr(model, 'feature_importances_'):
            log(f"{indent}-> Found feature_importances_")
            return model.feature_importances_
            
        # Case 2: Linear models (Fitted)
        if hasattr(model, 'coef_'):
            log(f"{indent}-> Found coef_")
            return np.abs(model.coef_[0])
            
        # Case 3: Calibrated Classifiers (Priority over generic .estimator)
        if hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
             log(f"{indent}-> Found calibrated_classifiers_ (len={len(model.calibrated_classifiers_)})")
             imps = []
             for i, clf in enumerate(model.calibrated_classifiers_):
                 log(f"{indent}  Calibrated classifier {i}: {type(clf)}")
                 
                 # Recurse into the component classifiers
                 # They usually have an 'estimator' attribute which IS fitted
                 if hasattr(clf, 'estimator'):
                     log(f"{indent}  -> Recursing into clf.estimator")
                     imp = get_model_importance_debug(clf.estimator, depth+2)
                     if imp is not None: imps.append(imp)
                 elif hasattr(clf, 'base_estimator'):
                     log(f"{indent}  -> Recursing into clf.base_estimator")
                     imp = get_model_importance_debug(clf.base_estimator, depth+2)
                     if imp is not None: imps.append(imp)
                 else:
                     log(f"{indent}  -> No estimator/base_estimator found on clf")
                     
             if imps:
                 log(f"{indent}-> Averaging {len(imps)} importances")
                 return np.mean(np.vstack(imps), axis=0)
             else:
                 log(f"{indent}-> No importances found in calibrated classifiers")

        # Case 4: Generic Wrappers (only if not calibrated)
        if hasattr(model, 'estimator'):
            log(f"{indent}-> Found .estimator, recursing... Type: {type(model.estimator)}")
            return get_model_importance_debug(model.estimator, depth+1)
            
        if hasattr(model, 'base_estimator'):
            log(f"{indent}-> Found .base_estimator, recursing... Type: {type(model.base_estimator)}")
            return get_model_importance_debug(model.base_estimator, depth+1)

    except Exception as e:
        log(f"{indent}Debug extraction error: {e}")
        log(traceback.format_exc())
        pass
    
    log(f"{indent}-> Failed to extract importance")
    return None

try:
    log(f"Loading model from {model_path}...")
    pipeline = load_model(model_path)
    
    model_step = pipeline.steps[-1][1]
    final_model = get_base_estimator(model_step)
    
    log("\\n--- Starting Extraction Debug ---")
    importances = get_model_importance_debug(final_model)
    
    if importances is not None:
        log(f"\\nSUCCESS: Extracted importances with shape {importances.shape}")
    else:
        log("\\nFAILURE: Could not extract importances.")

except Exception as e:
    log(f"Error checking model: {e}")
    log(traceback.format_exc())

finally:
    # Write log to file
    with open("debug_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_buffer))
