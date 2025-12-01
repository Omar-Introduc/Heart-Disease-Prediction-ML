import json
import os

nb_path = 'notebooks/02_Training_PyCaret.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_str = "".join(source)
        if "SAMPLE_FRAC = 0.01" in source_str:
            new_source = []
            for line in source:
                if "SAMPLE_FRAC = 0.01" in line:
                    new_source.append(line.replace("0.01", "0.05"))
                else:
                    new_source.append(line)
            cell['source'] = new_source
            found = True
            break

if found:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Successfully modified SAMPLE_FRAC.")
else:
    print("Could not find SAMPLE_FRAC definition.")
