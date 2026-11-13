"""Settings for the standalone, current-first DF/SD/DD experiment."""

import json
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parents[3]
DEFAULT_RESULTS_DIR = (
    REPO_ROOT / "logs/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_s4_1m"
)
CONFIG = json.loads((CODE_DIR / "experiment_config.json").read_text())
METHODS = ("df_min", "sd_iql", "dd_iql")
METHOD_IDS = dict(zip(METHODS, range(3)))
LABELS = {"df_min": "DF-IQL (min)", "sd_iql": "SD-IQL", "dd_iql": "DD-IQL"}
COLORS = {"df_min": "#cc79a7", "sd_iql": "#d55e00", "dd_iql": "#009e73"}


def cases():
    for mode in CONFIG["noise_modes"]:
        for method in METHODS:
            for delay in CONFIG["delays"]:
                for tau in CONFIG["tau_values"]:
                    yield 4, mode, method, delay, tau


def case_path(results_dir, case):
    _, mode, method, delay, tau = case
    return (
        Path(results_dir)
        / "raw/s4_a4"
        / mode
        / f"d{delay}"
        / f"tau{tau:g}"
        / f"{method}.npz"
    )


def noise_name(mode):
    return "IID" if mode.endswith("_iid") else "NN-like field"
