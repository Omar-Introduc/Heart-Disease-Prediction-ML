import json
import os

nb_path = 'notebooks/02_Training_PyCaret.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # source is a list of strings
        new_source = []
        changed_cell = False
        for line in source:
            if "SAMPLE_FRAC = 0.01" in line:
                print(f"Found line: {line.strip()}")
                new_line = line.replace("0.01", "0.10")
                print(f"Replaced with: {new_line.strip()}")
                new_source.append(new_line)
                changed_cell = True
            else:
                new_source.append(line)

        if changed_cell:
            cell['source'] = new_source
            found = True
            break

if found:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Successfully modified SAMPLE_FRAC to 0.10.")
else:
    print("Could not find SAMPLE_FRAC definition.")
