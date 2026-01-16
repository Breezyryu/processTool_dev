import json

# Read both notebooks  
with open('battery_analysis_v2_examples.ipynb', 'r', encoding='utf-8') as f:
    nb_working = json.load(f)

with open('battery_data_analysis_workflow.ipynb', 'r', encoding='utf-8') as f:
    nb_broken = json.load(f)

# Get working import cell
working_cells = [c for c in nb_working['cells'] if c['cell_type'] == 'code']
working_import = working_cells[0]['source']

# Get broken import cell  
broken_cells = [c for c in nb_broken['cells'] if c['cell_type'] == 'code']
broken_import = broken_cells[0]['source']

print("=== Working Import (battery_analysis_v2_examples.ipynb) ===")
for line in working_import:
    print(line, end='')

print("\n\n=== Current Import (battery_data_analysis_workflow.ipynb) ===")
for i, line in enumerate(broken_import):
    print(f"{i:2d}: {line}", end='')

# Fix the broken notebook by updating the first code cell
# Use the correct import structure
new_import_lines = [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from pathlib import Path\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# battery_analysis_v2 모듈 import\n",
    "from battery_analysis_v2.core.data_loader import (\n",
    "    detect_cycler_type,\n",
    "    PNELoader,\n",
    "    ToyoLoader,\n",
    "    CyclerType\n",
    ")\n",
    "\n",
    "from battery_analysis_v2.core.life_prediction.empirical import (\n",
    "    CapacityDegradationModel,\n",
    "    ApprovalLifePredictor,\n",
    "    ModelParameters,\n",
    "    FittingResult\n",
    ")\n",
    "\n",
    "# 그래프 스타일 설정\n",
    "plt.rcParams['figure.figsize'] = (14, 6)\n",
    "plt.rcParams['font.size'] = 10\n",
    "plt.rcParams['axes.grid'] = True\n",
    "plt.rcParams['grid.alpha'] = 0.3\n",
    "\n",
    "print('✓ 모듈 import 완료')"
]

# Update the notebook
nb_broken['cells'][2]['source'] = new_import_lines  # Cell index 2 is the first code cell

# Save the fixed notebook
with open('battery_data_analysis_workflow_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_broken, f, ensure_ascii=False, indent=1)

print("\n\n=== Fixed notebook saved to battery_data_analysis_workflow_fixed.ipynb ===")
print("Please review and replace the original file if correct.")
