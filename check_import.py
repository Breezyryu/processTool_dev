import json

# Read notebook
with open('battery_data_analysis_workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and print the first code cell
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
first_cell = code_cells[0]

print("=== First Code Cell (Full) ===")
for i, line in enumerate(first_cell['source']):
    print(f"{i:2d}: {line}", end='')
print()

# Look for empirical import section
print("\n=== Lines with 'empirical' or 'ModelParameters' ===")
for i, line in enumerate(first_cell['source']):
    if 'empirical' in line or 'ModelParameters' in line:
        print(f"{i:2d}: {line}", end='')
