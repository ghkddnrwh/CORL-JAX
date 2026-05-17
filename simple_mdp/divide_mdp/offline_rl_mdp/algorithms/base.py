from __future__ import annotations

import abc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np

from ..data import ReplayBuffer
from ..env import TwoPathBranchingMDP, evaluate_policy


@dataclass
class TrainResult:
    history: List[Dict[str, float]]


class BaseOfflineAlgorithm(abc.ABC):
    name: str = "base"

    def __init__(self, env: TwoPathBranchingMDP, gamma: float = 1.0, seed: int = 0):
        self.env = env
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.history: List[Dict[str, float]] = []

    @abc.abstractmethod
    def fit(self, replay_buffer: ReplayBuffer, optimal_q: np.ndarray | None = None) -> TrainResult:
        raise NotImplementedError

    @abc.abstractmethod
    def q_values(self) -> np.ndarray:
        """Returns array of shape [num_heads, num_states, num_actions]."""
        raise NotImplementedError

    @abc.abstractmethod
    def greedy_action(self, state: int) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def config_dict(self) -> Dict[str, object]:
        raise NotImplementedError

    def greedy_policy(self) -> np.ndarray:
        return np.asarray([self.greedy_action(s) for s in range(self.env.num_states)], dtype=np.int64)

    def mean_q(self) -> np.ndarray:
        return np.mean(self.q_values(), axis=0)

    def min_q(self) -> np.ndarray:
        return np.min(self.q_values(), axis=0)

    def q_for_comparison(self) -> np.ndarray:
        """Returns a single Q table [num_states, num_actions] used for diagnostics."""
        return self.mean_q()

    def q_error_metrics(self, optimal_q: np.ndarray) -> Dict[str, float]:
        current_q = self.q_for_comparison()
        if current_q.shape != optimal_q.shape:
            raise ValueError(
                f"shape mismatch between current_q {current_q.shape} and optimal_q {optimal_q.shape}"
            )

        valid_mask = np.zeros_like(current_q, dtype=bool)
        for state in range(self.env.num_states):
            valid_actions = self.env.valid_actions(state)
            valid_mask[state, valid_actions] = True

        diffs = current_q[valid_mask] - optimal_q[valid_mask]
        return {
            "q_mae_valid": float(np.mean(np.abs(diffs))),
            "q_rmse_valid": float(np.sqrt(np.mean(np.square(diffs)))),
            "q_max_abs_valid": float(np.max(np.abs(diffs))),
        }

    def evaluate_current_policy(self) -> Dict[str, float]:
        policy = self.greedy_policy()
        metrics = evaluate_policy(self.env, lambda s: int(policy[s]), gamma=self.gamma)
        return {
            "policy_return": float(metrics["discounted_return"]),
            "chose_short_path": float(1.0 if metrics["chose_short_path"] else 0.0),
            "selected_optimal_path": float(1.0 if metrics["selected_optimal_path"] else 0.0),
        }

    def minibatches(self, num_items: int, batch_size: int) -> Iterator[np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        order = self.rng.permutation(num_items)
        for start in range(0, num_items, batch_size):
            yield order[start : start + batch_size]

    def save(self, save_dir: str | Path) -> None:
        from ..utils import save_history_csv, save_json, save_numpy

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_json(save_dir / "algorithm_config.json", self.config_dict())
        save_history_csv(save_dir / "train_history.csv", self.history)
        save_numpy(save_dir / "q_tables.npy", self.q_values())
        save_numpy(save_dir / "mean_q.npy", self.mean_q())
        save_numpy(save_dir / "min_q.npy", self.min_q())
        save_json(save_dir / "greedy_policy.json", {"policy": self.greedy_policy().tolist()})


@dataclass
class TabularAlgoConfig:
    num_epochs: int = 500
    learning_rate: float = 0.1
    gamma: float = 1.0
    seed: int = 0
    batch_size: int = 1
    update_mode: str = "sample"  # sample or minibatch

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
