import unittest
import pandas as pd
from src.adapters import DataAdapter

class TestAdapter(unittest.TestCase):
    def test_mapping(self):
        feature_names = ['AGEG5YR', 'SEX', 'BMI5', 'SMOKE100', 'DIABETE4', 'EXERANY2']
        adapter = DataAdapter(feature_names)

        ui_input = {
            'Age': 30, # Should map to code 2 (25-29) or 3 (30-34)? 30 is inclusive?
            # My logic: <=29 is 2. <=34 is 3. So 30 is 3.
            'Sex': 'Male', # 1
            'BMI': 25.0, # 2500
            'Smoker': 'Yes', # 1
            'Diabetes': 'No', # 3
            'PhysicalActivity': 'Yes' # 1
        }

        df = adapter.map_ui_input_to_model(ui_input)

        self.assertEqual(df.iloc[0]['AGEG5YR'], 3)
        self.assertEqual(df.iloc[0]['SEX'], 1)
        self.assertEqual(df.iloc[0]['BMI5'], 2500)
        self.assertEqual(df.iloc[0]['SMOKE100'], 1)
        self.assertEqual(df.iloc[0]['DIABETE4'], 3)
        self.assertEqual(df.iloc[0]['EXERANY2'], 1)

if __name__ == '__main__':
    unittest.main()
