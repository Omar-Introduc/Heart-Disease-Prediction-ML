import pytest
import pandas as pd
from src.adapters import UserInputAdapter

def test_map_age_to_category():
    adapter = UserInputAdapter()
    assert adapter._map_age_to_category(20) == 1
    assert adapter._map_age_to_category(27) == 2
    assert adapter._map_age_to_category(85) == 13
    assert adapter._map_age_to_category(10) == 14

def test_transform_user_input():
    adapter = UserInputAdapter()
    user_input = {
        'Age': 30, # -> Cat 3
        'BMI': 25.0, # -> 2500
        'Smoker': 'Yes', # -> 1
        'Sex': 'Male', # -> 1
        'Diabetes': 'No', # -> 3
        'PhysicalActivity': 'Yes' # -> 1
    }

    # Transform
    df = adapter.transform(user_input)

    assert df.shape == (1, 6) # Ensure we have the mapped columns
    assert df['_AGEG5YR'].iloc[0] == 3
    assert df['_BMI5'].iloc[0] == 2500
    assert df['SMOKE100'].iloc[0] == 1
    assert df['SEXVAR'].iloc[0] == 1
    assert df['DIABETE4'].iloc[0] == 3
    assert df['EXERANY2'].iloc[0] == 1
