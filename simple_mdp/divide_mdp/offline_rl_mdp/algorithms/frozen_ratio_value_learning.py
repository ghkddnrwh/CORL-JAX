from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..data import ReplayBuffer
from .base import BaseOfflineAlgorithm, TabularAlgoConfig, TrainResult


@dataclass
class FrozenRatioValueLearningConfig(TabularAlgoConfig):
    noise_std: float = 0.0
    init_std: float = 1e-3
    fixed_update_freq: int = 10
    fixed_polyak_tau: float = 1.0
    value_learning_rate: float | None = None
    weight_temperature: float = 1.0
    min_update_weight: float = 0.0
    max_update_weight: float = 20.0


class FrozenRatioValueLearning(BaseOfflineAlgorithm):
    """Frozen Ratio Value Learning (FRVL).

    Maintains online Q(s, a), V(s), and lagged/frozen snapshots Q_f(s, a), V_f(s).

    Update rules:
      Q(s, a) <- Q(s, a) + lr_q [r + gamma V(s') - Q(s, a)]
      V(s)    <- V(s)    + lr_v * w_f(s, a) [Q(s, a) - V(s)]

    where the weighting term uses frozen values:
      w_f(s, a) = clip(exp(temperature * (Q_f(s, a) - V_f(s))), min_w, max_w)

    When V_f tracks max_a Q_f(s, a), the maximal action gets weight near 1 while
    suboptimal actions get weights below 1, which mimics action re-sampling without
    explicitly changing the replay buffer sampling distribution.
    """

    name = "frozen_ratio_value_learning"

    def __init__(self, env, config: FrozenRatioValueLearningConfig):
        super().__init__(env=env, gamma=config.gamma, seed=config.seed)
        if config.batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        if config.update_mode not in {"sample", "minibatch"}:
            raise ValueError("update_mode must be either 'sample' or 'minibatch'")
        if config.fixed_update_freq <= 0:
            raise ValueError("fixed_update_freq must be >= 1")
        if not (0.0 < config.fixed_polyak_tau <= 1.0):
            raise ValueError("fixed_polyak_tau must be in (0, 1]")
        if config.noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        if config.weight_temperature <= 0.0:
            raise ValueError("weight_temperature must be > 0")
        if config.max_update_weight < config.min_update_weight:
            raise ValueError("max_update_weight must be >= min_update_weight")

        self.config = config
        self.q_lr = float(config.learning_rate)
        self.v_lr = float(config.value_learning_rate if config.value_learning_rate is not None else config.learning_rate)

        self.q = self.rng.normal(
            loc=0.0,
            scale=config.init_std,
            size=(env.num_states, env.num_actions),
        )
        self.v = self.rng.normal(
            loc=0.0,
            scale=config.init_std,
            size=(env.num_states,),
        )
        self.q_f = self.q.copy()
        self.v_f = self.v.copy()
        self.total_updates = 0

    def fit(self, replay_buffer: ReplayBuffer, optimal_q: np.ndarray | None = None) -> TrainResult:
        transitions = list(replay_buffer.iter_transitions())

        for epoch in range(self.config.num_epochs):
            q_errors: List[float] = []
            v_errors: List[float] = []
            used_q_targets: List[float] = []
            used_weights: List[float] = []
            num_batches = 0

            if self.config.update_mode == "sample":
                for batch_indices in self.minibatches(len(transitions), batch_size=1):
                    idx = int(batch_indices[0])
                    tr = transitions[idx]

                    q_target = self._compute_q_target(tr.next_state, tr.reward, tr.done)
                    q_td = q_target - self.q[tr.state, tr.action]
                    self.q[tr.state, tr.action] += self.q_lr * q_td

                    weight = self._frozen_weight(tr.state, tr.action)
                    v_td = self.q[tr.state, tr.action] - self.v[tr.state]
                    self.v[tr.state] += self.v_lr * weight * v_td
                    self._inject_approximation_noise()

                    self.total_updates += 1
                    num_batches += 1
                    if self.total_updates % self.config.fixed_update_freq == 0:
                        self._update_frozen_tables()

                    q_errors.append(float(abs(q_td)))
                    v_errors.append(float(abs(v_td)))
                    used_q_targets.append(float(q_target))
                    used_weights.append(float(weight))
            else:
                for batch_indices in self.minibatches(len(transitions), self.config.batch_size):
                    batch = [transitions[idx] for idx in batch_indices]

                    q_targets = np.asarray(
                        [self._compute_q_target(tr.next_state, tr.reward, tr.done) for tr in batch],
                        dtype=np.float64,
                    )

                    q_updates = np.zeros_like(self.q)
                    for j, tr in enumerate(batch):
                        q_td = q_targets[j] - self.q[tr.state, tr.action]
                        q_updates[tr.state, tr.action] += self.q_lr * q_td
                        q_errors.append(float(abs(q_td)))
                        used_q_targets.append(float(q_targets[j]))

                    self.q += q_updates

                    v_updates = np.zeros_like(self.v)
                    for tr in batch:
                        weight = self._frozen_weight(tr.state, tr.action)
                        v_td = self.q[tr.state, tr.action] - self.v[tr.state]
                        v_updates[tr.state] += self.v_lr * weight * v_td
                        v_errors.append(float(abs(v_td)))
                        used_weights.append(float(weight))

                    self.v += v_updates
                    self._inject_approximation_noise()

                    self.total_updates += 1
                    num_batches += 1
                    if self.total_updates % self.config.fixed_update_freq == 0:
                        self._update_frozen_tables()

            eval_metrics = self.evaluate_current_policy()
            q_error_metrics = self.q_error_metrics(optimal_q) if optimal_q is not None else {}
            row = {
                "epoch": float(epoch),
                "num_batches": float(num_batches),
                "mean_abs_q_error": float(np.mean(q_errors)) if q_errors else 0.0,
                "mean_abs_v_error": float(np.mean(v_errors)) if v_errors else 0.0,
                "mean_q_target": float(np.mean(used_q_targets)) if used_q_targets else 0.0,
                "mean_update_weight": float(np.mean(used_weights)) if used_weights else 0.0,
                "mean_q": float(np.mean(self.q)),
                "max_q": float(np.max(self.q)),
                "min_q": float(np.min(self.q)),
                "mean_v": float(np.mean(self.v)),
                "max_v": float(np.max(self.v)),
                "min_v": float(np.min(self.v)),
                **q_error_metrics,
                **eval_metrics,
            }
            self.history.append(row)

        return TrainResult(history=self.history)

    def _compute_q_target(self, next_state: int, reward: float, done: bool) -> float:
        if done:
            return float(reward)
        return float(reward + self.gamma * self.v[next_state])

    def _frozen_weight(self, state: int, action: int) -> float:
        log_weight = self.config.weight_temperature * (self.q_f[state, action] - self.v_f[state])
        weight = float(np.exp(log_weight))
        return float(np.clip(weight, self.config.min_update_weight, self.config.max_update_weight))

    def _inject_approximation_noise(self) -> None:
        if self.config.noise_std <= 0.0:
            return
        self.q += self.rng.normal(loc=0.0, scale=self.config.noise_std, size=self.q.shape)
        self.v += self.rng.normal(loc=0.0, scale=self.config.noise_std, size=self.v.shape)

    def _update_frozen_tables(self) -> None:
        tau = self.config.fixed_polyak_tau
        self.q_f = tau * self.q + (1.0 - tau) * self.q_f
        self.v_f = tau * self.v + (1.0 - tau) * self.v_f

    def q_values(self) -> np.ndarray:
        return self.q[None, :, :].copy()

    def v_values(self) -> np.ndarray:
        return self.v.copy()

    def greedy_action(self, state: int) -> int:
        valid_actions = self.env.valid_actions(state)
        return int(valid_actions[np.argmax(self.q[state, valid_actions])])

    def config_dict(self) -> Dict[str, object]:
        return {
            "algorithm": self.name,
            "weighting": "exp(Q_f - V_f)",
            "fixed_snapshot_update": "polyak_or_periodic",
            **self.config.to_dict(),
            "effective_value_learning_rate": self.v_lr,
        }

    def save(self, save_dir: str | Path) -> None:
        super().save(save_dir)
        from ..utils import save_numpy

        save_dir = Path(save_dir)
        save_numpy(save_dir / "v_values.npy", self.v)
        save_numpy(save_dir / "fixed_q.npy", self.q_f)
        save_numpy(save_dir / "fixed_v.npy", self.v_f)
