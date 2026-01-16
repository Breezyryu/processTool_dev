import json

with open(r'battery_data_analysis_workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Total code cells: {len(code_cells)}\n")

# Find cells with ModelParameters
for i, cell in enumerate(code_cells):
    source = ''.join(cell['source'])
    if 'ModelParameters' in source:
        print(f"=== Cell {i} with ModelParameters ===")
        print(source[:800])
        print("\n")

# Also print first cell (imports)
print("=== First code cell (imports) ===")
print(''.join(code_cells[0]['source']))
