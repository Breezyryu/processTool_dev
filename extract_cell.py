import json

with open('battery_data_analysis_workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Get first code cell
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
first_cell_source = code_cells[0]['source']

# Write to a text file for easier viewing
with open('first_cell.txt', 'w', encoding='utf-8') as f:
    for i, line in enumerate(first_cell_source):
        f.write(f"{i:3d}: {line}")

print("Written to first_cell.txt")
print(f"Total lines: {len(first_cell_source)}")
