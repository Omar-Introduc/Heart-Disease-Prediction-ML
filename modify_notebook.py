import nbformat

def modify_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    modified_frac = False
    modified_iter = False

    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Modify SAMPLE_FRAC
            if 'SAMPLE_FRAC = ' in cell.source:
                lines = cell.source.split('\n')
                new_lines = []
                for line in lines:
                    if line.strip().startswith('SAMPLE_FRAC ='):
                        new_lines.append('SAMPLE_FRAC = 0.01  # Modified for quick test')
                        modified_frac = True
                    else:
                        new_lines.append(line)
                cell.source = '\n'.join(new_lines)

            # Modify n_iter in tune_model
            if 'tune_model(' in cell.source:
                # This is a bit more complex if it spans multiple lines, but looking at the read_file output:
                # tuned_model = tune_model(
                #    best_model,
                #    optimize='Recall',
                #    n_iter=50,
                #    choose_better=True,
                #    verbose=False
                # )
                # We can replace 'n_iter=50' with 'n_iter=2'
                if 'n_iter=50' in cell.source:
                    cell.source = cell.source.replace('n_iter=50', 'n_iter=2')
                    modified_iter = True

    if modified_frac and modified_iter:
        print("Successfully modified SAMPLE_FRAC and n_iter.")
    else:
        print(f"Modifications incomplete. FRAC: {modified_frac}, ITER: {modified_iter}")

    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    modify_notebook('notebooks/02_Training_PyCaret.ipynb')
