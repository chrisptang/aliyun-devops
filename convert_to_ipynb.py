import nbformat

# Read the generated Python code
with open("云效项目协作-BUG分析.ipynb", "r") as f:
    python_code = f.read()

# Create a new notebook
nb = nbformat.v4.new_notebook()

# Create a code cell with the Python code
code_cell = nbformat.v4.new_code_cell(python_code)

# Add the code cell to the notebook
nb['cells'].append(code_cell)

# Write the notebook to a new file
with open("云效项目协作-BUG分析.ipynb", "w") as f:
    nbformat.write(nb, f)