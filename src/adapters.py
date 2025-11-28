from typing import Any, Dict, List
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
                # We need raw_score=True to get probabilities
                predictions = predict_model(self.model, data=data, raw_score=True, verbose=False)

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


class UserInputAdapter:
    """
    Adapts human-readable dictionary inputs into the technical DataFrame structure
    expected by the model, applying necessary encodings and transformations.
    """
    def __init__(self):
        # We might not know the exact feature list at initialization,
        # but we can produce the known columns.
        # The model usually expects specific columns.
        pass

    def _map_age_to_category(self, age: int) -> int:
        """Maps age in years to BRFSS age category (AGEG5YR)."""
        if 18 <= age <= 24: return 1
        elif 25 <= age <= 29: return 2
        elif 30 <= age <= 34: return 3
        elif 35 <= age <= 39: return 4
        elif 40 <= age <= 44: return 5
        elif 45 <= age <= 49: return 6
        elif 50 <= age <= 54: return 7
        elif 55 <= age <= 59: return 8
        elif 60 <= age <= 64: return 9
        elif 65 <= age <= 69: return 10
        elif 70 <= age <= 74: return 11
        elif 75 <= age <= 79: return 12
        elif age >= 80: return 13
        else: return 14 # Missing or out of range

    def transform(self, user_input: Dict[str, Any]) -> pd.DataFrame:
        """
        Transforms user input dictionary into a DataFrame compatible with the model.
        Returns a DataFrame with the mapped features.
        """
        # Dictionary to hold the mapped values
        data = {}

        # Extract user inputs
        age = user_input.get('Age', 30)
        bmi = user_input.get('BMI', 25.0)
        smoker = user_input.get('Smoker', 'No')
        sex = user_input.get('Sex', 'Female')
        diabetes = user_input.get('Diabetes', 'No')
        phys_activity = user_input.get('PhysicalActivity', 'No')

        # Map inputs to model features (BRFSS Standards)

        # _AGEG5YR: Reported Age in 5 Year Bands
        data['_AGEG5YR'] = self._map_age_to_category(age)

        # _BMI5: Body Mass Index * 100
        data['_BMI5'] = int(bmi * 100)

        # SMOKE100: Smoked at least 100 cigarettes (1=Yes, 2=No)
        data['SMOKE100'] = 1 if smoker == 'Yes' else 2

        # SEXVAR: Sex of respondent (1=Male, 2=Female)
        data['SEXVAR'] = 1 if sex == 'Male' else 2

        # DIABETE4: (1=Yes, 3=No) - Simplified
        # In full dataset: 1=Yes, 2=Yes(preg), 3=No, 4=Pre-diabetes
        data['DIABETE4'] = 1 if diabetes == 'Yes' else 3

        # EXERANY2: Exercise in past 30 days (1=Yes, 2=No)
        data['EXERANY2'] = 1 if phys_activity == 'Yes' else 2

        # Additional features could be added here if the model expects them.
        # Ideally, we should pad with 0s or defaults for any other columns
        # the model might expect if they are not collected from UI.

        return pd.DataFrame([data])
