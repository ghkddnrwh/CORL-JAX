from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..data import ReplayBuffer
from .base import BaseOfflineAlgorithm, TabularAlgoConfig, TrainResult


@dataclass
class MonteCarloConfig(TabularAlgoConfig):
    num_q: int = 1
    init_std: float = 1e-3
    policy_extraction: str = "mean"  # mean or min


class MonteCarloReturnRegressor(BaseOfflineAlgorithm):
    name = "monte_carlo"

    def __init__(self, env, config: MonteCarloConfig):
        super().__init__(env=env, gamma=config.gamma, seed=config.seed)
        if config.num_q <= 0:
            raise ValueError("num_q must be >= 1")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        if config.update_mode not in {"sample", "minibatch"}:
            raise ValueError("update_mode must be either 'sample' or 'minibatch'")
        if config.policy_extraction not in {"mean", "min"}:
            raise ValueError("policy_extraction must be either 'mean' or 'min'")
        self.config = config
        self.q = self.rng.normal(
            loc=0.0,
            scale=config.init_std,
            size=(config.num_q, env.num_states, env.num_actions),
        )

    def fit(self, replay_buffer: ReplayBuffer, optimal_q: np.ndarray | None = None) -> TrainResult:
        transitions = list(replay_buffer.iter_transitions())
        mc_targets = replay_buffer.monte_carlo_targets(self.gamma)

        for epoch in range(self.config.num_epochs):
            errors: List[float] = []
            used_targets: List[float] = []
            num_batches = 0

            if self.config.update_mode == "sample":
                for batch_indices in self.minibatches(len(transitions), batch_size=1):
                    idx = int(batch_indices[0])
                    tr = transitions[idx]
                    target = float(mc_targets[idx])
                    error = target - self.q[:, tr.state, tr.action]
                    self.q[:, tr.state, tr.action] += self.config.learning_rate * error
                    num_batches += 1
                    errors.append(float(np.mean(np.abs(error))))
                    used_targets.append(target)
            else:
                for batch_indices in self.minibatches(len(transitions), self.config.batch_size):
                    batch_updates = np.zeros_like(self.q)
                    for idx in batch_indices:
                        tr = transitions[idx]
                        target = float(mc_targets[idx])
                        error = target - self.q[:, tr.state, tr.action]
                        batch_updates[:, tr.state, tr.action] += self.config.learning_rate * error
                        errors.append(float(np.mean(np.abs(error))))
                        used_targets.append(target)

                    self.q += batch_updates
                    num_batches += 1

            eval_metrics = self.evaluate_current_policy()
            q_error_metrics = self.q_error_metrics(optimal_q) if optimal_q is not None else {}
            self.history.append(
                {
                    "epoch": float(epoch),
                    "num_batches": float(num_batches),
                    "mean_abs_error": float(np.mean(errors)) if errors else 0.0,
                    "mean_target": float(np.mean(used_targets)) if used_targets else 0.0,
                    "mean_q": float(np.mean(self.q)),
                    "max_q": float(np.max(self.q)),
                    "min_q": float(np.min(self.q)),
                    **q_error_metrics,
                    **eval_metrics,
                }
            )

        return TrainResult(history=self.history)

    def q_values(self) -> np.ndarray:
        return self.q.copy()

    def q_for_comparison(self) -> np.ndarray:
        if self.config.policy_extraction == "min":
            return self.min_q()
        return self.mean_q()

    def greedy_action(self, state: int) -> int:
        valid_actions = self.env.valid_actions(state)
        if self.config.policy_extraction == "mean":
            action_values = np.mean(self.q[:, state, :], axis=0)
        else:
            action_values = np.min(self.q[:, state, :], axis=0)
        return int(valid_actions[np.argmax(action_values[valid_actions])])

    def config_dict(self) -> Dict[str, object]:
        return {
            "algorithm": self.name,
            **self.config.to_dict(),
        }
