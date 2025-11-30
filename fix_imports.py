import nbformat

def fix_numpy_import(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Find the cell with threshold optimization
    for cell in nb.cells:
        if "np.arange" in cell.source and "import numpy as np" not in cell.source:
            # Prepend the import
            cell.source = "import numpy as np\n" + cell.source
            print(f"Added 'import numpy as np' to cell in {notebook_path}")
            break

    # Also check the first import cell just in case we want it global
    # But local to the cell is fine and safe.

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    fix_numpy_import('notebooks/02_Training_PyCaret.ipynb')
