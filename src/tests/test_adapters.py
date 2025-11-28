import pytest
import pandas as pd
from src.adapters import transform_user_input, map_age_to_category

def test_map_age_to_category():
    assert map_age_to_category(20) == 1
    assert map_age_to_category(27) == 2
    assert map_age_to_category(85) == 13
    assert map_age_to_category(10) == 14

def test_transform_user_input():
    feature_names = ['_AGEG5YR', '_BMI5', 'SMOKE100', 'SEXVAR', 'DIABETE4', 'EXERANY2', 'OTHER_COL']
    user_input = {
        'Age': 30, # -> Cat 3
        'BMI': 25.0, # -> 2500
        'Smoker': 'Yes', # -> 1
        'Sex': 'Male', # -> 1
        'Diabetes': 'No', # -> 3
        'PhysicalActivity': 'Yes' # -> 1
    }

    df = transform_user_input(user_input, feature_names)

    assert df.shape == (1, 7)
    assert df['_AGEG5YR'].iloc[0] == 3
    assert df['_BMI5'].iloc[0] == 2500
    assert df['SMOKE100'].iloc[0] == 1
    assert df['SEXVAR'].iloc[0] == 1
    assert df['DIABETE4'].iloc[0] == 3
    assert df['EXERANY2'].iloc[0] == 1
    assert df['OTHER_COL'].iloc[0] == 0
