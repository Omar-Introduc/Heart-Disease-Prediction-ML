from typing import Protocol, runtime_checkable, Optional
import numpy as np
from pydantic import BaseModel, Field

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

class InputData(BaseModel):
    """
    Data contract for the Heart Disease Prediction Model (NHANES Schema).
    Validates input data before processing.
    """
    # Personal Data
    Age: int = Field(..., ge=18, le=100, description="Age in years")
    Sex: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    Height: float = Field(..., ge=130, le=220, description="Height in cm")

    # Vital Signs
    BMI: float = Field(..., ge=12.0, le=60.0, description="Body Mass Index")
    SystolicBP: float = Field(..., ge=80, le=220, description="Systolic Blood Pressure (mmHg)")
    DiastolicBP: Optional[float] = Field(None, ge=40, le=120, description="Diastolic Blood Pressure (mmHg)")
    WaistCircumference: float = Field(..., ge=50, le=180, description="Waist Circumference (cm)")

    # Biochemistry Profile
    TotalCholesterol: float = Field(..., ge=100, le=400, description="Total Cholesterol (mg/dL)")
    LDL: float = Field(..., ge=30, le=300, description="LDL Cholesterol (mg/dL)")
    Triglycerides: float = Field(..., ge=30, le=600, description="Triglycerides (mg/dL)")
    HbA1c: float = Field(..., ge=4.0, le=15.0, description="Glycated Hemoglobin (%)")
    Glucose: float = Field(..., ge=50, le=300, description="Fasting Glucose (mg/dL)")
    UricAcid: float = Field(..., ge=2.0, le=12.0, description="Uric Acid (mg/dL)")
    Creatinine: float = Field(..., ge=0.4, le=5.0, description="Serum Creatinine (mg/dL)")

    # Lifestyle / History
    Smoking: int = Field(..., ge=0, le=1, description="Smoked >100 cigarettes? (0/1)")
    Alcohol: int = Field(..., ge=0, le=1, description="Frequent alcohol consumption? (0/1)")
    PhysicalActivity: int = Field(..., ge=0, le=1, description="Vigorous physical activity? (0/1)")
    HealthInsurance: int = Field(..., ge=0, le=1, description="Has health insurance? (0/1)")
