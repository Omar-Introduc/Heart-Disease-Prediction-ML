from typing import Any
import pandas as pd
import numpy as np
from src.interfaces import HeartDiseaseModel

# Lazy import or safe import for PyCaret
try:
    from pycaret.classification import predict_model
except (ImportError, RuntimeError):
    # RuntimeError is raised by PyCaret on python 3.12
    predict_model = None

class PyCaretAdapter(HeartDiseaseModel):
    """
    Adapter to make PyCaret models compatible with HeartDiseaseModel protocol.
    """
    def __init__(self, model: Any):
        """
        :param model: The trained PyCaret pipeline/model.
        """
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        PyCaret models are usually pre-trained pipelines.
        Re-fitting might not be standard usage for the artifact.
        """
        raise NotImplementedError("PyCaretAdapter is intended for inference with pre-trained models.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probabilities for the positive class (1).
        """
        # Convert X to DataFrame if it's numpy
        if isinstance(X, np.ndarray):
            # Try to get feature names from the underlying steps
            feature_names = None
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = self.model.feature_names_in_

            if feature_names is not None:
                data = pd.DataFrame(X, columns=feature_names)
            else:
                data = pd.DataFrame(X)
        else:
            data = X

        if predict_model:
            try:
                # predict_model in PyCaret usually returns a DF with scores
                predictions = predict_model(self.model, data=data, raw_score=True)

                # Check for various score column naming conventions
                if 'prediction_score_1' in predictions.columns:
                    return predictions['prediction_score_1'].values
                elif 'Score_1' in predictions.columns:
                     return predictions['Score_1'].values
                elif 'prediction_score' in predictions.columns:
                    # If binary and label is present, we might need to infer
                    label = predictions['prediction_label']
                    score = predictions['prediction_score']
                    # If label is 1, prob is score. If label is 0, prob is 1-score.
                    # This assumes 'prediction_score' is the probability of the *predicted* class
                    return np.where(label == 1, score, 1 - score)
                else:
                    return self.model.predict_proba(data)[:, 1]
            except Exception:
                 return self.model.predict_proba(data)[:, 1]
        else:
            # Fallback to direct sklearn call if PyCaret not available
            return self.model.predict_proba(data)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels with custom threshold.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
