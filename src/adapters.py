from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from src.interfaces import HeartDiseaseModel, InputData
from pydantic import ValidationError
try:
    from pycaret.classification import predict_model
except (ImportError, RuntimeError):
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
        if isinstance(X, np.ndarray):
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
                predictions = predict_model(self.model, data=data, raw_score=True, verbose=False)
                if 'prediction_score_1' in predictions.columns:
                    return predictions['prediction_score_1'].values
                elif 'Score_1' in predictions.columns:
                     return predictions['Score_1'].values
                elif 'prediction_score' in predictions.columns:
                    label = predictions['prediction_label']
                    score = predictions['prediction_score']
                    return np.where(label == 1, score, 1 - score)
                else:
                    return self.model.predict_proba(data)[:, 1]
            except Exception:
                 return self.model.predict_proba(data)[:, 1]
        else:
            return self.model.predict_proba(data)[:, 1]
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels with custom threshold.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
class UserInputAdapter:
    """
    Adapts human-readable dictionary inputs into the technical DataFrame structure
    expected by the model, applying necessary validations and type conversions.
    """
    def __init__(self):
        pass
    def transform(self, user_input: Dict[str, Any]) -> pd.DataFrame:
        """
        Transforms user input dictionary into a DataFrame compatible with the model.
        Validates input using Pydantic InputData model.
        Args:
            user_input (Dict[str, Any]): Dictionary containing user inputs.
        Returns:
            pd.DataFrame: A DataFrame with the columns expected by the model.
        Raises:
            ValidationError: If input data is invalid.
        """
        try:
            validated_data = InputData(**user_input)
        except ValidationError as e:
            raise ValueError(f"Invalid input data: {e}")
        data_dict = validated_data.dict()
        if data_dict.get('DiastolicBP') is None:
            data_dict['DiastolicBP'] = 80.0 
        df = pd.DataFrame([data_dict])
        return df
