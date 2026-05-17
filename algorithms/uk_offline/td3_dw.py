import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-expert-v2"
    seed: int = 0
    eval_freq: int = int(25e3)
    n_episodes: int = 10
    max_timesteps: int = int(1e6)
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    # Dataset
    buffer_size: int = 2_000_000
    batch_size: int = 256
    normalize: bool = True
    normalize_reward: bool = False
    # Value learning
    discount: float = 0.99
    tau: float = 0.005
    qf_lr: float = 3e-4
    vf_lr: float = 3e-4
    delayed_update_period: int = 250
    min_weight_exponent: float = 0.0
    max_weight_exponent: float = 2.0
    weight_logit_clip: float = 10.0
    beta_min: float = 0.0
    # Policy fitting during evaluation
    actor_lr: float = 3e-4
    eval_actor_steps: int = 1000
    eval_actor_batch_size: int = 256
    eval_actor_eval_freq: int = 2000
    final_eval_actor_steps: int = 50000
    reset_actor_on_eval: bool = False
    alpha: float = 2.5  # Coefficient for Q function in actor loss
    # Logging
    project: str = "ORL-BIAS"
    group: str = "TD3-DW"
    name: str = "TD3_DW"
    log_wandb: bool = True
    log_every: int = 100

    def __post_init__(self):
        self.project = "ORL-BIAS"
        self.group = "TD3-DW-SMALL-REWARD"
        self.name = "TD3-DW"

        assert self.min_weight_exponent >= 0.0, "min_weight_exponent must be >= 0"
        assert self.max_weight_exponent >= 0.0, "max_weight_exponent must be >= 0"
        assert self.beta_min >= 0.0, "beta_min must be >= 0"
        assert (
            self.max_weight_exponent >= self.min_weight_exponent
        ), "max_weight_exponent must be >= min_weight_exponent"

        self.name = f"{self.name}-{self.env}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name, str(self.seed))


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def is_scalar_value(value: Any) -> bool:
    if isinstance(value, (int, float, bool, np.number)):
        return True
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return True
    return False


def to_python_scalar(value: Any) -> Union[int, float, bool]:
    if isinstance(value, np.ndarray):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def prefix_log_keys(log: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}/{key}": value for key, value in log.items()}


def save_logs_npz(logs: List[Dict[str, Any]], path: str) -> None:
    if len(logs) == 0:
        return

    keys = logs[0].keys()
    data_to_save: Dict[str, np.ndarray] = {}
    for key in keys:
        values = [log[key] for log in logs]
        try:
            data_to_save[key] = np.asarray(values)
        except ValueError:
            data_to_save[key] = np.asarray(values, dtype=object)

    np.savez(path, **data_to_save)


def save_and_upload_eval_logs(
    eval_logs: List[Dict[str, Any]],
    checkpoints_path: Optional[str],
    log_wandb: bool,
):
    if checkpoints_path is None or len(eval_logs) == 0:
        return

    eval_logs_path = os.path.join(checkpoints_path, "eval_logs.npz")
    save_logs_npz(eval_logs, eval_logs_path)

    if log_wandb and wandb.run is not None:
        wandb.save(eval_logs_path, policy="now")


def normalize_episode_scores(env: gym.Env, eval_scores: np.ndarray) -> np.ndarray:
    return np.asarray(
        [env.get_normalized_score(float(score)) * 100.0 for score in eval_scores],
        dtype=np.float32,
    )


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        # dataset["rewards"] -= 1.0
        dataset["rewards"] *= 100


class MLP(nn.Module):
    def __init__(self, dims: List[int], squeeze_output: bool = False):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dimension must be 1 when squeeze_output=True")
            self.net = nn.Sequential(*layers)
            self.squeeze_output = True
        else:
            self.net = nn.Sequential(*layers)
            self.squeeze_output = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.squeeze_output:
            return x.squeeze(-1)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class QFunction(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.q = MLP([state_dim + action_dim, 256, 256, 1], squeeze_output=True)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=1)
        return self.q(sa)


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.v = MLP([state_dim, 256, 256, 1], squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class TD3DelayedWeighting:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        max_steps: int,
        discount: float = 0.99,
        tau: float = 0.005,
        delayed_update_period: int = 250,
        min_weight_exponent: float = 0.0,
        max_weight_exponent: float = 2.0,
        weight_logit_clip: float = 10.0,
        beta_min: float = 0.0,
        alpha: float = 2.5,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.actor_reset_state = copy.deepcopy(actor.state_dict())
        self.actor_optimizer_reset_state = copy.deepcopy(actor_optimizer.state_dict())

        self.qf = q_network
        self.q_target = copy.deepcopy(q_network).requires_grad_(False).to(device)
        self.q_delayed = copy.deepcopy(self.q_target).requires_grad_(False).to(device)
        self.vf = v_network
        self.v_target = copy.deepcopy(v_network).requires_grad_(False).to(device)
        self.v_delayed = copy.deepcopy(self.v_target).requires_grad_(False).to(device)

        self.q_optimizer = q_optimizer
        self.v_optimizer = v_optimizer

        self.discount = discount
        self.tau = tau
        self.delayed_update_period = delayed_update_period
        self.min_weight_exponent = min_weight_exponent
        self.max_weight_exponent = max_weight_exponent
        self.weight_logit_clip = weight_logit_clip
        self.beta_min = beta_min
        self.max_steps = max_steps
        self.alpha = alpha

        self.total_it = 0
        self.device = device

    def _current_weight_exponent(self) -> float:
        progress = min(float(self.total_it) / max(float(self.max_steps), 1.0), 1.0)
        return self.min_weight_exponent + (
            self.max_weight_exponent - self.min_weight_exponent
        ) * progress

    def _compute_beta(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            delayed_q = self.q_delayed(states, actions)
            delayed_v = self.v_delayed(states)
            raw_delayed_adv = delayed_q - delayed_v
            delayed_adv = raw_delayed_adv.clamp(
                min=-self.weight_logit_clip, max=self.weight_logit_clip
            )
            exponent = self._current_weight_exponent()
            beta = torch.where(
                delayed_adv < 0.0,
                torch.exp(exponent * delayed_adv),
                torch.ones_like(delayed_adv),
            )
            beta = torch.clamp(beta, min=self.beta_min)

            adv_stats: Dict[str, float] = {
                "delayed_adv_mean": delayed_adv.mean().item(),
                "delayed_adv_min": delayed_adv.min().item(),
                "delayed_adv_max": delayed_adv.max().item(),
                "raw_delayed_adv_mean": raw_delayed_adv.mean().item(),
                "raw_delayed_adv_min": raw_delayed_adv.min().item(),
                "raw_delayed_adv_max": raw_delayed_adv.max().item(),
                "clipped_low_frac": (
                    raw_delayed_adv <= -self.weight_logit_clip
                ).float().mean().item(),
                "clipped_high_frac": (
                    raw_delayed_adv >= self.weight_logit_clip
                ).float().mean().item(),
                "negative_adv_frac": (raw_delayed_adv < 0.0).float().mean().item(),
            }
        return beta, adv_stats

    def _update_q(
        self,
        next_observations: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        log_dict: Dict[str, float],
    ):
        with torch.no_grad():
            next_v = self.v_target(next_observations)
            target_q = rewards + (1.0 - dones.float()) * self.discount * next_v

        q = self.qf(observations, actions)
        q_loss = F.mse_loss(q, target_q)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        log_dict["q_loss"] = q_loss.item()
        log_dict["q_mean"] = q.mean().item()
        log_dict["target_q_mean"] = target_q.mean().item()

    def _update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict[str, float],
    ):
        beta, adv_stats = self._compute_beta(observations, actions)
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        value_residual = v - target_q
        value_loss = (beta * value_residual.pow(2)).mean()
        self.v_optimizer.zero_grad()
        value_loss.backward()
        self.v_optimizer.step()

        log_dict["value_loss"] = value_loss.item()
        log_dict["beta_mean"] = beta.mean().item()
        log_dict["beta_min"] = beta.min().item()
        log_dict["beta_max"] = beta.max().item()
        log_dict["weight_exponent"] = self._current_weight_exponent()
        log_dict["v_mean"] = v.mean().item()
        log_dict.update(adv_stats)

    def _update_targets(self):
        soft_update(self.q_target, self.qf, self.tau)
        soft_update(self.v_target, self.vf, self.tau)
        if self.total_it % self.delayed_update_period == 0:
            hard_update(self.q_delayed, self.q_target)
            hard_update(self.v_delayed, self.v_target)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        observations, actions, rewards, next_observations, dones = batch
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)
        log_dict: Dict[str, float] = {}

        self._update_q(next_observations, observations, actions, rewards, dones, log_dict)
        self._update_v(observations, actions, log_dict)
        self._update_targets()

        return log_dict

    def reset_actor(self):
        self.actor.load_state_dict(self.actor_reset_state)
        self.actor_optimizer.load_state_dict(self.actor_optimizer_reset_state)

    def fit_actor(
        self,
        replay_buffer: ReplayBuffer,
        steps: int,
        batch_size: int,
        eval_env: Optional[gym.Env] = None,
        eval_episodes: int = 0,
        eval_seed: int = 0,
        eval_interval: int = 0,
    ) -> Dict[str, Any]:
        eval_fit_log: Dict[str, Any] = {
            "fit_actor/final_loss": np.nan,
            "fit_actor/final_q": np.nan,
            "fit_actor/final_score_mean": np.nan,
            "fit_actor/final_score_std": np.nan,
            "fit_actor/final_d4rl_normalized_score_mean": np.nan,
            "fit_actor/final_d4rl_normalized_score_std": np.nan,
            "fit_actor/best_score_mean": np.nan,
            "fit_actor/best_score_std": np.nan,
            "fit_actor/best_d4rl_normalized_score_mean": np.nan,
            "fit_actor/best_d4rl_normalized_score_std": np.nan,
            "fit_actor/inner_eval_steps": [],
            "fit_actor/inner_score_mean": [],
            "fit_actor/inner_score_std": [],
            "fit_actor/inner_d4rl_normalized_score_mean": [],
            "fit_actor/inner_d4rl_normalized_score_std": [],
        }
        if steps <= 0:
            return eval_fit_log

        self.actor.train()
        best_normalized_score_mean = -np.inf

        for fit_step in range(1, steps + 1):
            observations, action, _, _, _ = replay_buffer.sample(batch_size)
            observations = observations.to(self.device)
            action = action.to(self.device)

            pi = self.actor(observations)
            q = self.qf(observations, pi)
            lmbda = self.alpha / q.abs().mean().detach().clamp(min=1e-6)
            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            eval_fit_log["fit_actor/final_loss"] = float(actor_loss.item())
            eval_fit_log["fit_actor/final_q"] = float(q.mean().item())

            should_eval = (
                eval_env is not None
                and eval_episodes > 0
                and eval_interval > 0
                and (fit_step % eval_interval == 0 or fit_step == steps)
            )
            if should_eval:
                eval_scores = eval_actor(
                    eval_env,
                    self.actor,
                    device=self.device,
                    n_episodes=eval_episodes,
                    seed=eval_seed,
                )
                normalized_eval_scores = normalize_episode_scores(eval_env, eval_scores)

                eval_score_mean = float(np.mean(eval_scores))
                eval_score_std = float(np.std(eval_scores))
                normalized_eval_score_mean = float(np.mean(normalized_eval_scores))
                normalized_eval_score_std = float(np.std(normalized_eval_scores))

                eval_fit_log["fit_actor/inner_eval_steps"].append(int(fit_step))
                eval_fit_log["fit_actor/inner_score_mean"].append(eval_score_mean)
                eval_fit_log["fit_actor/inner_score_std"].append(eval_score_std)
                eval_fit_log["fit_actor/inner_d4rl_normalized_score_mean"].append(
                    normalized_eval_score_mean
                )
                eval_fit_log["fit_actor/inner_d4rl_normalized_score_std"].append(
                    normalized_eval_score_std
                )
                eval_fit_log["fit_actor/final_score_mean"] = eval_score_mean
                eval_fit_log["fit_actor/final_score_std"] = eval_score_std
                eval_fit_log[
                    "fit_actor/final_d4rl_normalized_score_mean"
                ] = normalized_eval_score_mean
                eval_fit_log[
                    "fit_actor/final_d4rl_normalized_score_std"
                ] = normalized_eval_score_std

                if normalized_eval_score_mean > best_normalized_score_mean:
                    best_normalized_score_mean = normalized_eval_score_mean
                    eval_fit_log["fit_actor/best_score_mean"] = eval_score_mean
                    eval_fit_log["fit_actor/best_score_std"] = eval_score_std
                    eval_fit_log[
                        "fit_actor/best_d4rl_normalized_score_mean"
                    ] = normalized_eval_score_mean
                    eval_fit_log[
                        "fit_actor/best_d4rl_normalized_score_std"
                    ] = normalized_eval_score_std

                print(
                    f"[fit_actor] step {fit_step}/{steps}: "
                    f"eval_mean={eval_score_mean:.3f}, eval_std={eval_score_std:.3f}, "
                    f"D4RL_mean={normalized_eval_score_mean:.3f}, "
                    f"D4RL_std={normalized_eval_score_std:.3f}"
                )

        return eval_fit_log


    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_target": self.q_target.state_dict(),
            "q_delayed": self.q_delayed.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_target": self.v_target.state_dict(),
            "v_delayed": self.v_delayed.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_reset_state": self.actor_reset_state,
            "actor_optimizer_reset_state": self.actor_optimizer_reset_state,
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_target.load_state_dict(state_dict["q_target"])
        self.q_delayed.load_state_dict(state_dict["q_delayed"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])

        self.vf.load_state_dict(state_dict["vf"])
        self.v_target.load_state_dict(state_dict["v_target"])
        self.v_delayed.load_state_dict(state_dict["v_delayed"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_reset_state = state_dict.get(
            "actor_reset_state", copy.deepcopy(self.actor.state_dict())
        )
        self.actor_optimizer_reset_state = state_dict.get(
            "actor_optimizer_reset_state",
            copy.deepcopy(self.actor_optimizer.state_dict()),
        )

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)

        config_path = os.path.join(config.checkpoints_path, "config.yaml")
        if os.path.exists(config_path):
            print(f"Error: The file '{config_path}' already exists.")
            exit(1)

        with open(config_path, "w") as f:
            pyrallis.dump(config, f)

    seed = config.seed
    set_seed(seed, env)

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    q_network = QFunction(state_dim, action_dim).to(config.device)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)

    v_network = ValueFunction(state_dim).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "max_steps": config.max_timesteps,
        "discount": config.discount,
        "tau": config.tau,
        "delayed_update_period": config.delayed_update_period,
        "min_weight_exponent": config.min_weight_exponent,
        "max_weight_exponent": config.max_weight_exponent,
        "weight_logit_clip": config.weight_logit_clip,
        "beta_min": config.beta_min,
        "device": config.device,
        "alpha": config.alpha,
    }

    print("---------------------------------------")
    print(f"Training TD3-DW, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    trainer = TD3DelayedWeighting(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file, map_location=config.device))
        actor = trainer.actor

    if config.log_wandb:
        wandb_init(asdict(config))

    eval_logs: List[Dict[str, Any]] = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)

        if config.log_wandb and (t + 1) % config.log_every == 0:
            wandb.log(log_dict, step=trainer.total_it)

        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            if config.reset_actor_on_eval:
                trainer.reset_actor()

            eval_fit_log = trainer.fit_actor(
                replay_buffer=replay_buffer,
                steps=config.eval_actor_steps,
                batch_size=config.eval_actor_batch_size,
                eval_env=env,
                eval_episodes=config.n_episodes,
                eval_seed=config.seed,
                eval_interval=config.eval_actor_eval_freq,
            )

            eval_log: Dict[str, Any] = {
                "timestep": int(t + 1),
                "eval/reward_mean": eval_fit_log["fit_actor/final_score_mean"],
                "eval/reward_std": eval_fit_log["fit_actor/final_score_std"],
                "eval/normalized_score_mean": eval_fit_log[
                    "fit_actor/final_d4rl_normalized_score_mean"
                ],
                "eval/normalized_score_std": eval_fit_log[
                    "fit_actor/final_d4rl_normalized_score_std"
                ],
            }
            eval_log.update(eval_fit_log)
            eval_logs.append(eval_log.copy())

            if config.log_wandb:
                wandb_eval_log = {
                    key: to_python_scalar(value)
                    for key, value in eval_log.items()
                    if is_scalar_value(value)
                }
                wandb.log(wandb_eval_log, step=trainer.total_it)

            save_and_upload_eval_logs(
                eval_logs=eval_logs,
                checkpoints_path=config.checkpoints_path,
                log_wandb=config.log_wandb,
            )

    final_fit_log: Optional[Dict[str, Any]] = None
    final_actor = Actor(state_dim, action_dim, max_action).to(config.device)
    final_actor_optimizer = torch.optim.Adam(final_actor.parameters(), lr=config.actor_lr)
    final_fit_trainer = copy.deepcopy(trainer)
    final_fit_trainer.actor = final_actor
    final_fit_trainer.actor_optimizer = final_actor_optimizer
    final_fit_trainer.actor_reset_state = copy.deepcopy(final_actor.state_dict())
    final_fit_trainer.actor_optimizer_reset_state = copy.deepcopy(final_actor_optimizer.state_dict())
    final_fit_trainer.reset_actor()

    print("---------------------------------------")
    print("Post-training fresh actor fitting starts")
    print("---------------------------------------")
    final_fit_log_raw = final_fit_trainer.fit_actor(
        replay_buffer=replay_buffer,
        steps=config.final_eval_actor_steps,
        batch_size=config.eval_actor_batch_size,
        eval_env=env,
        eval_episodes=config.n_episodes,
        eval_seed=config.seed,
        eval_interval=config.eval_actor_eval_freq,
    )
    final_fit_log = {"timestep": int(config.max_timesteps)}
    final_fit_log.update(prefix_log_keys(final_fit_log_raw, "post_training"))

    if config.log_wandb:
        wandb_final_fit_log = {
            key: to_python_scalar(value)
            for key, value in final_fit_log.items()
            if is_scalar_value(value)
        }
        wandb.log(wandb_final_fit_log, step=trainer.total_it)

    if config.checkpoints_path is not None:
        torch.save(
            trainer.state_dict(),
            os.path.join(config.checkpoints_path, "checkpoint.pt"),
        )
        torch.save(
            final_fit_trainer.actor.state_dict(),
            os.path.join(config.checkpoints_path, "post_training_actor.pt"),
        )

        if final_fit_log is not None:
            save_logs_npz(
                [final_fit_log],
                os.path.join(config.checkpoints_path, "post_training_fit_log.npz"),
            )

        if config.log_wandb and wandb.run is not None:
            wandb.save(os.path.join(config.checkpoints_path, "checkpoint.pt"), policy="now")
            wandb.save(
                os.path.join(config.checkpoints_path, "post_training_actor.pt"),
                policy="now",
            )
            if final_fit_log is not None:
                wandb.save(
                    os.path.join(config.checkpoints_path, "post_training_fit_log.npz"),
                    policy="now",
                )

        save_and_upload_eval_logs(
            eval_logs=eval_logs,
            checkpoints_path=config.checkpoints_path,
            log_wandb=config.log_wandb,
        )


if __name__ == "__main__":
    train()
