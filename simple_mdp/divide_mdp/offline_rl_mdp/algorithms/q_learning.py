from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..data import ReplayBuffer
from .base import BaseOfflineAlgorithm, TabularAlgoConfig, TrainResult


@dataclass
class ClippedDoubleQConfig(TabularAlgoConfig):
    num_q: int = 2
    noise_std: float = 0.0
    target_update_freq: int = 1
    polyak_tau: float = 1.0
    init_std: float = 1e-3
    policy_extraction: str = "mean"  # mean or min


class ClippedDoubleQEnsemble(BaseOfflineAlgorithm):
    name = "clipped_q_learning"

    def __init__(self, env, config: ClippedDoubleQConfig):
        super().__init__(env=env, gamma=config.gamma, seed=config.seed)
        if config.num_q <= 0:
            raise ValueError("num_q must be >= 1")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        if config.update_mode not in {"sample", "minibatch"}:
            raise ValueError("update_mode must be either 'sample' or 'minibatch'")
        if config.target_update_freq <= 0:
            raise ValueError("target_update_freq must be >= 1")
        if not (0.0 < config.polyak_tau <= 1.0):
            raise ValueError("polyak_tau must be in (0, 1]")
        if config.noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        if config.policy_extraction not in {"mean", "min"}:
            raise ValueError("policy_extraction must be either 'mean' or 'min'")

        self.config = config
        self.online_q = self.rng.normal(
            loc=0.0,
            scale=config.init_std,
            size=(config.num_q, env.num_states, env.num_actions),
        )
        self.target_q = self.online_q.copy()
        self.total_updates = 0

    def fit(self, replay_buffer: ReplayBuffer, optimal_q: np.ndarray | None = None) -> TrainResult:
        transitions = list(replay_buffer.iter_transitions())

        for epoch in range(self.config.num_epochs):
            td_errors: List[float] = []
            targets: List[float] = []
            num_batches = 0

            if self.config.update_mode == "sample":
                for batch_indices in self.minibatches(len(transitions), batch_size=1):
                    idx = int(batch_indices[0])
                    tr = transitions[idx]
                    target = self._compute_target(tr.next_state, tr.reward, tr.done)
                    td = target - self.online_q[:, tr.state, tr.action]
                    self.online_q[:, tr.state, tr.action] += self.config.learning_rate * td
                    self._inject_approximation_noise()

                    self.total_updates += 1
                    num_batches += 1
                    if self.total_updates % self.config.target_update_freq == 0:
                        self._update_target_network()

                    td_errors.append(float(np.mean(np.abs(td))))
                    targets.append(float(target))
            else:
                for batch_indices in self.minibatches(len(transitions), self.config.batch_size):
                    batch = [transitions[idx] for idx in batch_indices]
                    batch_targets = np.asarray(
                        [self._compute_target(tr.next_state, tr.reward, tr.done) for tr in batch],
                        dtype=np.float64,
                    )

                    batch_updates = np.zeros_like(self.online_q)
                    for j, tr in enumerate(batch):
                        td = batch_targets[j] - self.online_q[:, tr.state, tr.action]
                        batch_updates[:, tr.state, tr.action] += self.config.learning_rate * td
                        td_errors.append(float(np.mean(np.abs(td))))
                        targets.append(float(batch_targets[j]))

                    self.online_q += batch_updates
                    self._inject_approximation_noise()

                    self.total_updates += 1
                    num_batches += 1
                    if self.total_updates % self.config.target_update_freq == 0:
                        self._update_target_network()

            eval_metrics = self.evaluate_current_policy()
            q_error_metrics = self.q_error_metrics(optimal_q) if optimal_q is not None else {}
            row = {
                "epoch": float(epoch),
                "num_batches": float(num_batches),
                "mean_abs_td_error": float(np.mean(td_errors)) if td_errors else 0.0,
                "mean_target": float(np.mean(targets)) if targets else 0.0,
                "mean_q": float(np.mean(self.online_q)),
                "max_q": float(np.max(self.online_q)),
                "min_q": float(np.min(self.online_q)),
                **q_error_metrics,
                **eval_metrics,
            }
            self.history.append(row)

        return TrainResult(history=self.history)

    def _compute_target(self, next_state: int, reward: float, done: bool) -> float:
        if done:
            return float(reward)

        valid_next_actions = self.env.valid_actions(next_state)
        clipped_next_q = np.min(self.target_q[:, next_state, :], axis=0)
        next_v = float(np.max(clipped_next_q[valid_next_actions]))
        return float(reward + self.gamma * next_v)

    def _inject_approximation_noise(self) -> None:
        if self.config.noise_std <= 0.0:
            return
        noise = self.rng.normal(
            loc=0.0,
            scale=self.config.noise_std,
            size=self.online_q.shape,
        )
        self.online_q += noise

    def _update_target_network(self) -> None:
        tau = self.config.polyak_tau
        self.target_q = tau * self.online_q + (1.0 - tau) * self.target_q

    def q_values(self) -> np.ndarray:
        return self.online_q.copy()

    def q_for_comparison(self) -> np.ndarray:
        if self.config.policy_extraction == "min":
            return self.min_q()
        return self.mean_q()

    def greedy_action(self, state: int) -> int:
        valid_actions = self.env.valid_actions(state)
        if self.config.policy_extraction == "mean":
            action_values = np.mean(self.online_q[:, state, :], axis=0)
        else:
            action_values = np.min(self.online_q[:, state, :], axis=0)
        return int(valid_actions[np.argmax(action_values[valid_actions])])

    def config_dict(self) -> Dict[str, object]:
        return {
            "algorithm": self.name,
            "noise_application": "post_minibatch_global_q_table",
            **self.config.to_dict(),
        }
