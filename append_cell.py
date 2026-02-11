import nbformat as nbf

nb_path = "/Users/yining/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/code/analysis_20251113.ipynb"

# Read the notebook
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Create a new code cell
code = """# Check unique values for each column
# (Added by Antigravity)
for col in data0.columns:
    print(f"--- {col} ---")
    print(data0[col].unique())
    print("\\n")
"""

new_cell = nbf.v4.new_code_cell(code)

# Append the cell
nb.cells.append(new_cell)

# Write back
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Successfully appended new cell to notebook.")
