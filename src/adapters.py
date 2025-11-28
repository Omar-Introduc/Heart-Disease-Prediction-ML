import pandas as pd
import numpy as np
import json

class DataAdapter:
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def map_ui_input_to_model(self, ui_input: dict) -> pd.DataFrame:
        """
        Maps user inputs (human readable) to the model's expected feature vector.
        Unspecified features are filled with defaults (usually 0 or median).
        """
        # Initialize with zeros or NaNs
        # Ridge Classifier handles 0s fine.
        data = {col: 0 for col in self.feature_names}

        # --- Mappings ---

        # Age: 18-99 -> AGEG5YR (1-13)
        # 1: 18-24, 2: 25-29, ..., 13: 80+
        age = ui_input.get('Age', 30)
        if age <= 24: age_code = 1
        elif age <= 29: age_code = 2
        elif age <= 34: age_code = 3
        elif age <= 39: age_code = 4
        elif age <= 44: age_code = 5
        elif age <= 49: age_code = 6
        elif age <= 54: age_code = 7
        elif age <= 59: age_code = 8
        elif age <= 64: age_code = 9
        elif age <= 69: age_code = 10
        elif age <= 74: age_code = 11
        elif age <= 79: age_code = 12
        else: age_code = 13

        data['AGEG5YR'] = age_code
        data['_AGEG5YR'] = age_code # Handle both if present (ingestion strips _, but just in case)

        # Sex: Male/Female -> SEX (1/2) or SEXVAR (1/2)
        sex = ui_input.get('Sex', 'Male')
        sex_code = 1 if sex == 'Male' else 2
        data['SEX'] = sex_code
        data['SEXVAR'] = sex_code
        data['_SEX'] = sex_code

        # BMI: 10.0-60.0 -> BMI5 (int, *100)
        bmi = ui_input.get('BMI', 25.0)
        data['BMI5'] = int(bmi * 100)
        data['_BMI5'] = int(bmi * 100)

        # Smoker: Yes/No -> SMOKE100 (1=Yes, 2=No)
        # Also SMOKER3 (1=Every day, 2=Some days, 3=Former, 4=Never)
        smoker = ui_input.get('Smoker', 'No')
        if smoker == 'Yes':
            data['SMOKE100'] = 1
            data['_SMOKER3'] = 1 # Rough approx
            data['SMOKER3'] = 1
        else:
            data['SMOKE100'] = 2
            data['_SMOKER3'] = 4
            data['SMOKER3'] = 4

        # Diabetes: Yes/No -> DIABETE4 (1=Yes, 3=No)
        diabetes = ui_input.get('Diabetes', 'No')
        if diabetes == 'Yes':
            data['DIABETE4'] = 1
        else:
            data['DIABETE4'] = 3 # 3 is No

        # Physical Activity: Yes/No -> EXERANY2 (1=Yes, 2=No)
        phys = ui_input.get('PhysicalActivity', 'No')
        if phys == 'Yes':
            data['EXERANY2'] = 1
            data['_TOTINDA'] = 1 # Calculated var
        else:
            data['EXERANY2'] = 2
            data['_TOTINDA'] = 2

        # Convert to DataFrame
        # Filter keys to only those in feature_names to avoid errors if model is strict
        filtered_data = {k: v for k, v in data.items() if k in self.feature_names}

        # If any feature from feature_names is missing (should be covered by init), it's 0
        return pd.DataFrame([filtered_data])

def get_adapter(feature_names_path='models/feature_names.json'):
    try:
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        return DataAdapter(feature_names)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        return None
