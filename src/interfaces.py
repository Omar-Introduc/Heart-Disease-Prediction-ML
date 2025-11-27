from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class HeartDiseaseModel(Protocol):
    """
    Protocol defining the interface for Heart Disease Prediction models.
    Both the scratch implementation and the production (PyCaret) wrapper must adhere to this.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.
        """
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        Returns a 1D array of probabilities for the positive class (Heart Disease).
        """
        ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        Allows specifying a custom threshold to optimize Recall/Precision.
        """
        ...
