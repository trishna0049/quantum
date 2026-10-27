"""Execute notebook code cells sequentially for local validation."""

import json
import os
from pathlib import Path


project_root = Path(__file__).resolve().parent
os.chdir(project_root)
notebook_path = project_root / "notebooks" / "qft_vs_fft.ipynb"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}

for index, cell in enumerate(notebook["cells"], start=1):
    if cell["cell_type"] == "code":
        print(f"Executing code cell {index}")
        source = "".join(cell["source"])
        exec(compile(source, f"{notebook_path}:cell-{index}", "exec"), namespace)

print("Notebook code cells completed successfully.")
