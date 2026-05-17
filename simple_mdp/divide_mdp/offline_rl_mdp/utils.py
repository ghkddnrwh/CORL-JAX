from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def save_json(path: str | Path, obj: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_numpy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def save_history_csv(path: str | Path, rows: Iterable[Dict[str, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_param_string(params: Dict[str, object]) -> str:
    pieces = []
    for key in sorted(params):
        value = params[key]
        safe_value = str(value).replace("/", "-")
        pieces.append(f"{key}={safe_value}")
    return "__".join(pieces) if pieces else "default"


def plot_q_error_history(
    path: str | Path,
    history: List[Dict[str, float]],
    metric_key: str = "q_rmse_valid",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        return

    epochs = [row["epoch"] for row in history if "epoch" in row and metric_key in row]
    metric_values = [row[metric_key] for row in history if "epoch" in row and metric_key in row]
    if not epochs:
        return

    label_map = {
        "q_rmse_valid": "Q RMSE vs optimal Q*",
        "q_mae_valid": "Q MAE vs optimal Q*",
        "q_max_abs_valid": "Q max abs error vs optimal Q*",
    }
    ylabel = label_map.get(metric_key, metric_key)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, metric_values, label=ylabel)
    plt.axhline(0.0, linestyle="--", label="Perfect match = 0")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Distance between learned Q and optimal Q* during training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_probability_curve(
    path: str | Path,
    epochs: List[float],
    probabilities: List[float],
    ylabel: str,
    title: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not epochs:
        return

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, probabilities, label=ylabel)
    plt.axhline(1.0, linestyle="--", label="Perfect selection = 1.0")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Epoch")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
