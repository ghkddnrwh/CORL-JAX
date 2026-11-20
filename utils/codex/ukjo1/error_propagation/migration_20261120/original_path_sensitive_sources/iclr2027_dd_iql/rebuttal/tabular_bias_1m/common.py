"""Paths, settings and the six algorithms in the rebuttal experiment."""

import json
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "logs/iclr2027_dd_iql/rebuttal/tabular_bias_1m"
CONFIG = json.loads((CODE_DIR / "experiment_config.json").read_text())
BASELINES = ("max_min", "double_q")
IQL = ("df_min", "df_mean", "sd_iql", "dd_iql")
METHODS = BASELINES + IQL
LABELS = {
    "max_min": "Min Q",
    "double_q": "Double Q",
    "df_min": "DF min",
    "df_mean": "DF mean",
    "sd_iql": "SD-IQL",
    "dd_iql": "DD-IQL",
}
COLORS = {
    "max_min": "#888888",
    "double_q": "#56a5d8",
    "df_min": "#cc79a7",
    "df_mean": "#9467bd",
    "sd_iql": "#d55e00",
    "dd_iql": "#009e73",
}


def cases(sizes=None, noise_modes=None, methods=None):
    for size in sizes or CONFIG["sizes"]:
        for mode in noise_modes or CONFIG["noise_modes"]:
            for method in methods or METHODS:
                if method in BASELINES:
                    yield size, mode, method, None, None
                else:
                    for delay in CONFIG["delays"]:
                        for tau in CONFIG["tau_values"]:
                            yield size, mode, method, delay, tau


def case_path(results_dir, case):
    size, mode, method, delay, tau = case
    base = Path(results_dir) / "raw" / f"s{size}_a{size}" / mode
    if method in BASELINES:
        return base / "baselines" / f"{method}.npz"
    return base / f"d{delay}" / f"tau{tau:g}" / f"{method}.npz"


def noise_name(mode):
    return "IID" if mode.endswith("_iid") else "NN-like field"
