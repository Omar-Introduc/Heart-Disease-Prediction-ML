import unittest
import pandas as pd
from src.adapters import UserInputAdapter
from src.interfaces import InputData
from pydantic import ValidationError

class TestUserInputAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = UserInputAdapter()
        self.valid_input = {
            'Age': 45,
            'Sex': 1,
            'Height': 175.0,
            'BMI': 24.5,
            'SystolicBP': 120.0,
            'DiastolicBP': 80.0,
            'WaistCircumference': 90.0,
            'TotalCholesterol': 200.0,
            'LDL': 100.0,
            'Triglycerides': 150.0,
            'HbA1c': 5.5,
            'Glucose': 90.0,
            'UricAcid': 5.0,
            'Creatinine': 0.9,
            'Smoking': 0,
            'Alcohol': 0,
            'PhysicalActivity': 1,
            'HealthInsurance': 1
        }

    def test_transform_valid_input(self):
        """Test transformation with valid inputs."""
        df = self.adapter.transform(self.valid_input)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df['Age'][0], 45)
        self.assertEqual(df['SystolicBP'][0], 120.0)

    def test_missing_optional_field(self):
        """Test missing optional DiastolicBP is handled."""
        input_data = self.valid_input.copy()
        del input_data['DiastolicBP'] # Remove optional

        # Pydantic should accept it (Optional), Adapter should fill default
        df = self.adapter.transform(input_data)
        self.assertEqual(df['DiastolicBP'][0], 80.0)

    def test_invalid_range_raises_error(self):
        """Test validation error for out of range values."""
        input_data = self.valid_input.copy()
        input_data['Age'] = 150 # Invalid

        with self.assertRaises(ValueError): # Adapter raises ValueError which wraps ValidationError
            self.adapter.transform(input_data)

    def test_invalid_type_raises_error(self):
        input_data = self.valid_input.copy()
        input_data['Age'] = "Not a number"
        with self.assertRaises(ValueError):
            self.adapter.transform(input_data)

if __name__ == '__main__':
    unittest.main()
