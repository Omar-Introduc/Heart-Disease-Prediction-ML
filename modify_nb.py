import json
import os

nb_path = 'notebooks/02_Training_PyCaret.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The replacement source code lines
new_config_block = [
    "# Load Schema Config (REPLACED WITH MANUAL DEFINITION)\n",
    "# with open(CONFIG_PATH, 'r') as f:\n",
    "#     config = json.load(f)\n",
    "\n",
    "# Define manually the columns based on EDA and Extraction\n",
    "numeric_features = [\n",
    "    'Age', 'SystolicBP', 'BMI', 'WaistCircumference', 'Height', \n",
    "    'TotalCholesterol', 'Triglycerides', 'LDL', 'HbA1c', 'Glucose', \n",
    "    'Creatinine', 'UricAcid', 'ALT_Enzyme', 'Albumin', 'Potassium', \n",
    "    'Sodium', 'GGT_Enzyme', 'AST_Enzyme', 'IncomeRatio'\n",
    "]\n",
    "categorical_features = [\n",
    "    'Sex', 'Race', 'Education', 'Smoking', 'Alcohol', \n",
    "    'PhysicalActivity', 'HealthInsurance'\n",
    "]\n",
    "target = 'HeartDisease'\n",
    "\n",
    "features = numeric_features + categorical_features\n"
]

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_str = "".join(source)
        if "with open(CONFIG_PATH, 'r') as f:" in source_str and "config = json.load(f)" in source_str:
            # Construct the full new source for that cell.
            full_new_source = [
                "# ==========================================\n",
                "# 1. LOAD DATA\n",
                "# ==========================================\n",
                "if not os.path.exists(DATA_PATH):\n",
                "    raise FileNotFoundError(f\"Data file not found at {DATA_PATH}\")\n",
                "\n",
                "df = pd.read_parquet(DATA_PATH)\n",
                "print(f\"Original Data Shape: {df.shape}\")\n",
                "\n"
            ] + new_config_block + [
                "\n",
                "# Filter only relevant columns\n",
                "df = df[features + [target]]\n",
                "\n",
                "if SAMPLE_FRAC < 1.0:\n",
                "    df = df.sample(frac=SAMPLE_FRAC, random_state=42)\n",
                "    print(f\"Sampled Data Shape: {df.shape}\")\n",
                "else:\n",
                "    print(\"Using Full Dataset\")\n"
            ]

            cell['source'] = full_new_source
            found = True
            break

if found:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Successfully modified the notebook.")
else:
    print("Could not find the target cell.")
