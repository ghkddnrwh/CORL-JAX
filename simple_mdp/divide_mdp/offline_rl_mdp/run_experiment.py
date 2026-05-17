from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    repo_root_script = Path(__file__).resolve().parent.parent / "run_experiment.py"
    runpy.run_path(str(repo_root_script), run_name="__main__")
