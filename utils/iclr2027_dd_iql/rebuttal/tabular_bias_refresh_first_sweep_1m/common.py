"""Settings for the standalone, current-first DF/SD/DD experiment."""

import json
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parents[3]
DEFAULT_RESULTS_DIR = (
    REPO_ROOT / "logs/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sizes_1m"
)
CONFIG = json.loads((CODE_DIR / "experiment_config.json").read_text())
IQL = ("df_min", "sd_iql", "dd_iql")
METHODS = ("double_q",) + IQL
METHOD_IDS = {"df_min": 0, "sd_iql": 1, "dd_iql": 2, "double_q": 3}
LABELS = {
    "max_min": "Clipped Double Q",
    "double_q": "Double Q",
    "df_min": "DF-IQL",
    "sd_iql": "SD-IQL",
    "dd_iql": "DD-IQL",
}
COLORS = {
    "max_min": "#777777",
    "double_q": "#56a5d8",
    "df_min": "#cc79a7",
    "sd_iql": "#d55e00",
    "dd_iql": "#009e73",
}


def cases(sizes=None):
    for size in sizes or CONFIG["sizes"]:
        for mode in CONFIG["noise_modes"]:
            yield size, mode, "double_q", None, None
            for method in IQL:
                for delay in CONFIG["delays"]:
                    for tau in CONFIG["tau_values"]:
                        yield size, mode, method, delay, tau


def case_path(results_dir, case):
    size, mode, method, delay, tau = case
    base = Path(results_dir) / "raw" / f"s{size}_a{size}" / mode
    if method == "double_q":
        return base / "baselines/double_q.npz"
    return base / f"d{delay}" / f"tau{tau:g}" / f"{method}.npz"


def noise_name(mode):
    return "IID" if mode.endswith("_iid") else "NN-like field"
