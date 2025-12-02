import json
import os

notebook_path = 'notebooks/06_Ethics_Analysis.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Modify imports
# Cell index 1 (0-based) is the imports cell
import_cell = nb['cells'][1]
new_source = []
for line in import_cell['source']:
    if 'import pickle' in line:
        new_source.append("from pycaret.classification import load_model\n")
    else:
        new_source.append(line)
import_cell['source'] = new_source

# 2. Modify model loading
# Cell index 3 is the loading cell
load_cell = nb['cells'][3]
new_load_source = [
    "# Load Model\n",
    "model_path = '../models/best_pipeline'\n",
    "try:\n",
    "    pipeline = load_model(model_path)\n",
    "    print(\"Model loaded successfully.\")\n",
    "except Exception as e:\n",
    "    print(f\"Model not found or error loading at {model_path}: {e}\")\n",
    "    # Fallback for demo if model missing in sandbox\n",
    "    from sklearn.dummy import DummyClassifier\n",
    "    pipeline = DummyClassifier(strategy='most_frequent')\n",
    "    pipeline.fit(np.zeros((10, 10)), np.zeros(10))\n",
    "\n",
    "# Load Data\n",
    "data_path = '../data/02_intermediate/process_data.parquet'\n",
    "try:\n",
    "    df = pd.read_parquet(data_path)\n",
    "    print(f\"Data loaded. Shape: {df.shape}\")\n",
    "except FileNotFoundError:\n",
    "    print(\"Data file not found. Creating synthetic data.\")\n",
    "    # Synthetic fallback\n",
    "    df = pd.DataFrame({\n",
    "        'Age': np.random.randint(20, 80, 200),\n",
    "        'Sex': np.random.choice(['Male', 'Female'], 200),\n",
    "        'SystolicBP': np.random.randint(90, 180, 200),\n",
    "        'BMI': np.random.uniform(18, 40, 200),\n",
    "        'HeartDisease': np.random.randint(0, 2, 200)\n",
    "    })\n",
    "\n",
    "# Ensure target exists\n",
    "target = 'HeartDisease'\n",
    "if target not in df.columns:\n",
    "    # Try to find target or use last col\n",
    "    target = df.columns[-1]"
]
load_cell['source'] = new_load_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
