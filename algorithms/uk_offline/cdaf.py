import copy
import os
import random
import sys
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
import yaml

TensorBatch = Dict[str, torch.Tensor]

ALGORITHM_NAME = "CDAF"
ALGORITHM_FULL_NAME = "Conservative Delayed Advantage Filtering"


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
    hyperparams_path: Optional[str] = "hyperparams/cdaf_jax.yml"
    use_hyperparams: bool = True

    # Dataset
    buffer_size: int = 2_000_000
    batch_size: int = 256
    normalize: bool = True
    normalize_reward: bool = False

    # CDAF value learning
    discount: float = 0.99
    tau: float = 0.005
    qf_lr: float = 3e-4
    vf_lr: float = 3e-4
    delayed_update_period: int = 250
    min_weight_exponent: float = 0.0
    max_weight_exponent: float = 2.0
    weight_logit_clip: float = 10.0
    beta_min: float = 0.0

    # TD3+BC actor fitting during evaluation
    actor_lr: float = 3e-4
    eval_actor_steps: int = 1000
    eval_actor_batch_size: int = 256
    eval_actor_eval_freq: int = 2000
    reset_actor_on_eval: bool = False
    alpha: float = 2.5
    bc_coef: float = 1.0

    # Standalone actor refit from a saved checkpoint.
    # Used when --load_model is provided with --max_timesteps 0.
    refit_actor_steps: int = 50000
    actor_refit_dir_name: str = "actor_refit"

    # Logging
    project: str = "ORL-BIAS"
    group: str = "CDAF"
    name: str = "CDAF"
    log_wandb: bool = True
    log_every: int = 100

    def __post_init__(self):
        refresh_algorithm_names(self)
        validate_config(self)


def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = "ORL-BIAS"
    config.group = ALGORITHM_NAME
    config.name = f"{ALGORITHM_NAME}-{config.env}"


def validate_config(config: TrainConfig) -> None:
    assert config.min_weight_exponent >= 0.0, "min_weight_exponent must be >= 0"
    assert config.max_weight_exponent >= 0.0, "max_weight_exponent must be >= 0"
    assert config.beta_min >= 0.0, "beta_min must be >= 0"
    assert config.max_weight_exponent >= config.min_weight_exponent
    assert config.delayed_update_period > 0
    assert config.bc_coef >= 0.0
    assert config.batch_size > 0
    assert config.eval_actor_batch_size > 0
    assert config.eval_actor_eval_freq > 0
    assert config.refit_actor_steps >= 0
    assert config.actor_refit_dir_name != ""


def _cli_overridden_fields(argv: Optional[List[str]] = None) -> set:
    argv = sys.argv[1:] if argv is None else argv
    overridden = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0].replace("-", "_")
        if key:
            overridden.add(key)
    return overridden


def _coerce_hparam_value(value: Any) -> Any:
    # YAML often loads !!float 1e6 as float, but step counts should be ints in this script.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def apply_env_hyperparams(config: TrainConfig) -> TrainConfig:
    """Load RL-Zoo-style env-specific hyperparameters and merge them into config.

    Priority is:
        dataclass defaults < hyperparams YAML < explicit CLI flags

    Supported alias:
        n_timesteps -> max_timesteps
    """
    if not config.use_hyperparams or config.hyperparams_path is None:
        refresh_algorithm_names(config)
        validate_config(config)
        return config

    hparam_path = Path(config.hyperparams_path)
    if not hparam_path.exists():
        print(f"Hyperparameter file not found: {hparam_path}. Using dataclass/CLI values.")
        refresh_algorithm_names(config)
        validate_config(config)
        return config

    with open(hparam_path, "r") as f:
        all_hyperparams = yaml.safe_load(f) or {}

    if config.env not in all_hyperparams:
        print(f"No hyperparameters found for env '{config.env}' in {hparam_path}. Using dataclass/CLI values.")
        refresh_algorithm_names(config)
        validate_config(config)
        return config

    env_hyperparams = all_hyperparams[config.env] or {}
    cli_overrides = _cli_overridden_fields()
    aliases = {
        "n_timesteps": "max_timesteps",
    }
    config_fields = set(TrainConfig.__dataclass_fields__.keys())
    applied, skipped_unknown, skipped_cli = [], [], []

    for raw_key, raw_value in env_hyperparams.items():
        key = aliases.get(raw_key, raw_key)
        if key not in config_fields:
            skipped_unknown.append(raw_key)
            continue
        if key in cli_overrides or raw_key in cli_overrides:
            skipped_cli.append(raw_key)
            continue
        setattr(config, key, _coerce_hparam_value(raw_value))
        applied.append(f"{raw_key}->{key}" if raw_key != key else key)

    refresh_algorithm_names(config)
    validate_config(config)

    if applied:
        print(f"Loaded hyperparameters for {config.env} from {hparam_path}: {', '.join(applied)}")
    if skipped_cli:
        print(f"Kept CLI overrides for: {', '.join(skipped_cli)}")
    if skipped_unknown:
        print(f"Ignored unknown hyperparameter keys for CDAF: {', '.join(skipped_unknown)}")
    return config


def finalize_checkpoint_path(config: TrainConfig) -> TrainConfig:
    if config.checkpoints_path is not None:
        config.checkpoints_path = os.path.join(config.checkpoints_path, config.name, str(config.seed))
    return config


def select_torch_device(device: str) -> torch.device:
    requested = device.lower()
    if requested == "gpu":
        requested = "cuda"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested PyTorch device '{device}' is not available. Falling back to cpu.")
        requested = "cpu"
    torch_device = torch.device(requested)
    print(f"Using PyTorch device: {torch_device}")
    return torch_device


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)


def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: Union[np.ndarray, float], std: Union[np.ndarray, float]):
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
        device: Union[str, torch.device],
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self._next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self._device = device

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")

        self._states[:n_transitions] = data["observations"].astype(np.float32)
        self._actions[:n_transitions] = data["actions"].astype(np.float32)
        self._rewards[:n_transitions] = data["rewards"][..., None].astype(np.float32)
        self._next_states[:n_transitions] = data["next_observations"].astype(np.float32)
        self._dones[:n_transitions] = data["terminals"][..., None].astype(np.float32)
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        batch = {
            "observations": self._states[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_observations": self._next_states[indices],
            "dones": self._dones[indices],
        }
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=self._device)
            for key, value in batch.items()
        }


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
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
    if torch.is_tensor(value) and value.ndim == 0:
        return True
    return False


def to_python_scalar(value: Any) -> Union[int, float, bool]:
    if isinstance(value, np.ndarray):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().item()
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
        # Preserves the reward preprocessing in the original main training code.
        dataset["rewards"] *= 100.0


class MLP(nn.Module):
    def __init__(self, dims: List[int], squeeze_output: bool = False):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if squeeze_output and dims[-1] != 1:
            raise ValueError("Last dimension must be 1 when squeeze_output=True")
        self.net = nn.Sequential(*layers)
        self.squeeze_output = squeeze_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.squeeze_output:
            return x.squeeze(-1)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: Union[str, torch.device] = "cpu") -> np.ndarray:
        state_t = torch.as_tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state_t).cpu().numpy().flatten()


class QFunction(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q = MLP([state_dim + action_dim, hidden_dim, hidden_dim, 1], squeeze_output=True)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q(sa)


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.v = MLP([state_dim, hidden_dim, hidden_dim, 1], squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


@dataclass
class ActorState:
    actor: nn.Module
    optimizer: torch.optim.Optimizer


class CDAF:
    """Conservative Delayed Advantage Filtering (CDAF) for offline RL.

    Main training learns Q and V with delayed negative-advantage filtering.
    Policy extraction is performed with TD3+BC-style actor fitting.
    """

    def __init__(
        self,
        max_action: float,
        state_dim: int,
        action_dim: int,
        max_steps: int,
        qf_lr: float = 3e-4,
        vf_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        delayed_update_period: int = 250,
        min_weight_exponent: float = 0.0,
        max_weight_exponent: float = 2.0,
        weight_logit_clip: float = 10.0,
        beta_min: float = 0.0,
        alpha: float = 2.5,
        bc_coef: float = 1.0,
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
    ):
        torch.manual_seed(seed)
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.discount = discount
        self.tau = tau
        self.delayed_update_period = delayed_update_period
        self.min_weight_exponent = min_weight_exponent
        self.max_weight_exponent = max_weight_exponent
        self.weight_logit_clip = weight_logit_clip
        self.beta_min = beta_min
        self.alpha = alpha
        self.bc_coef = bc_coef
        self.actor_lr = actor_lr
        self.device = torch.device(device)

        actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action).to(self.device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)

        self.qf = QFunction(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(self.device)
        self.q_delayed = copy.deepcopy(self.q_target).requires_grad_(False).to(self.device)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=qf_lr)

        self.vf = ValueFunction(state_dim=state_dim).to(self.device)
        self.v_target = copy.deepcopy(self.vf).requires_grad_(False).to(self.device)
        self.v_delayed = copy.deepcopy(self.v_target).requires_grad_(False).to(self.device)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=vf_lr)

        self.initial_actor_params = copy.deepcopy(actor.state_dict())
        self.initial_actor_opt_state = copy.deepcopy(actor_optimizer.state_dict())
        self.actor_state = ActorState(actor=actor, optimizer=actor_optimizer)

        self.total_it = 0

    def _current_weight_exponent(self) -> float:
        progress = min(float(self.total_it) / max(float(self.max_steps), 1.0), 1.0)
        return self.min_weight_exponent + (
            self.max_weight_exponent - self.min_weight_exponent
        ) * progress

    def _compute_beta(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            delayed_q = self.q_delayed(observations, actions)
            delayed_v = self.v_delayed(observations)
            raw_delayed_adv = delayed_q - delayed_v
            delayed_adv = raw_delayed_adv.clamp(
                min=-self.weight_logit_clip,
                max=self.weight_logit_clip,
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
                "clipped_low_frac": (raw_delayed_adv <= -self.weight_logit_clip).float().mean().item(),
                "clipped_high_frac": (raw_delayed_adv >= self.weight_logit_clip).float().mean().item(),
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
            target_v_q = self.q_target(observations, actions)

        v = self.vf(observations)
        value_residual = v - target_v_q
        value_loss = (beta * value_residual.pow(2)).mean()

        self.v_optimizer.zero_grad()
        value_loss.backward()
        self.v_optimizer.step()

        log_dict["value_loss"] = value_loss.item()
        log_dict["v_mean"] = v.mean().item()
        log_dict["target_v_q_mean"] = target_v_q.mean().item()
        log_dict["beta_mean"] = beta.mean().item()
        log_dict["beta_min"] = beta.min().item()
        log_dict["beta_max"] = beta.max().item()
        log_dict["weight_exponent"] = self._current_weight_exponent()
        log_dict.update(adv_stats)

    def _update_targets(self):
        soft_update(self.q_target, self.qf, self.tau)
        soft_update(self.v_target, self.vf, self.tau)
        if self.total_it % self.delayed_update_period == 0:
            hard_update(self.q_delayed, self.q_target)
            hard_update(self.v_delayed, self.v_target)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"].squeeze(-1)
        next_observations = batch["next_observations"]
        dones = batch["dones"].squeeze(-1)

        log_dict: Dict[str, float] = {}
        self._update_q(next_observations, observations, actions, rewards, dones, log_dict)
        self._update_v(observations, actions, log_dict)
        self._update_targets()
        return log_dict

    def _new_initial_actor_state(self) -> ActorState:
        actor = Actor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
        ).to(self.device)
        optimizer = torch.optim.Adam(actor.parameters(), lr=self.actor_lr)
        actor.load_state_dict(copy.deepcopy(self.initial_actor_params))
        optimizer.load_state_dict(copy.deepcopy(self.initial_actor_opt_state))
        return ActorState(actor=actor, optimizer=optimizer)

    def reset_actor(self):
        self.actor_state.actor.load_state_dict(copy.deepcopy(self.initial_actor_params))
        self.actor_state.optimizer.load_state_dict(copy.deepcopy(self.initial_actor_opt_state))

    @torch.no_grad()
    def actor_act(self, actor_state: ActorState, state: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
        action = actor_state.actor(state_t)
        return action.cpu().numpy()[0]

    @torch.no_grad()
    def eval_actor(
        self,
        env: gym.Env,
        actor_state: ActorState,
        n_episodes: int,
        seed: int,
    ) -> np.ndarray:
        env.seed(seed)
        was_training = actor_state.actor.training
        actor_state.actor.eval()
        episode_rewards = []
        for _ in range(n_episodes):
            state, done = env.reset(), False
            episode_reward = 0.0
            while not done:
                action = self.actor_act(actor_state, state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)
        if was_training:
            actor_state.actor.train()
        return np.asarray(episode_rewards, dtype=np.float32)

    def fit_actor(
        self,
        replay_buffer: ReplayBuffer,
        actor_state: ActorState,
        steps: int,
        batch_size: int,
        eval_env: Optional[gym.Env] = None,
        eval_episodes: int = 0,
        eval_seed: int = 0,
        eval_interval: int = 0,
        prefix: str = "fit_actor",
    ) -> Tuple[ActorState, Dict[str, Any]]:
        eval_fit_log: Dict[str, Any] = {
            f"{prefix}/final_loss": np.nan,
            f"{prefix}/final_q": np.nan,
            f"{prefix}/final_lambda": np.nan,
            f"{prefix}/final_bc_loss": np.nan,
            f"{prefix}/final_score_mean": np.nan,
            f"{prefix}/final_score_std": np.nan,
            f"{prefix}/final_d4rl_normalized_score_mean": np.nan,
            f"{prefix}/final_d4rl_normalized_score_std": np.nan,
            f"{prefix}/best_score_mean": np.nan,
            f"{prefix}/best_score_std": np.nan,
            f"{prefix}/best_d4rl_normalized_score_mean": np.nan,
            f"{prefix}/best_d4rl_normalized_score_std": np.nan,
            f"{prefix}/inner_eval_steps": [],
            f"{prefix}/inner_score_mean": [],
            f"{prefix}/inner_score_std": [],
            f"{prefix}/inner_d4rl_normalized_score_mean": [],
            f"{prefix}/inner_d4rl_normalized_score_std": [],
        }
        if steps <= 0:
            return actor_state, eval_fit_log

        actor_state.actor.train()
        q_requires_grad = [param.requires_grad for param in self.qf.parameters()]
        for param in self.qf.parameters():
            param.requires_grad_(False)
        best_normalized_score_mean = -np.inf

        try:
            for fit_step in range(1, steps + 1):
                batch = replay_buffer.sample(batch_size)
                observations = batch["observations"]
                actions = batch["actions"]

                pi = actor_state.actor(observations)
                q = self.qf(observations, pi)
                lmbda = self.alpha / q.abs().mean().detach().clamp(min=1e-6)
                bc_loss = F.mse_loss(pi, actions)
                actor_loss = -lmbda * q.mean() + self.bc_coef * bc_loss

                actor_state.optimizer.zero_grad()
                actor_loss.backward()
                actor_state.optimizer.step()

                step_log = {
                    "loss": actor_loss.item(),
                    "q_mean": q.mean().item(),
                    "lambda": lmbda.item(),
                    "bc_loss": bc_loss.item(),
                    "bc_coef": self.bc_coef,
                }
                eval_fit_log[f"{prefix}/final_loss"] = step_log["loss"]
                eval_fit_log[f"{prefix}/final_q"] = step_log["q_mean"]
                eval_fit_log[f"{prefix}/final_lambda"] = step_log["lambda"]
                eval_fit_log[f"{prefix}/final_bc_loss"] = step_log["bc_loss"]

                should_eval = (
                    eval_env is not None
                    and eval_episodes > 0
                    and eval_interval > 0
                    and (fit_step % eval_interval == 0 or fit_step == steps)
                )
                if should_eval:
                    eval_scores = self.eval_actor(
                        eval_env,
                        actor_state,
                        n_episodes=eval_episodes,
                        seed=eval_seed,
                    )
                    normalized_eval_scores = normalize_episode_scores(eval_env, eval_scores)

                    eval_score_mean = float(np.mean(eval_scores))
                    eval_score_std = float(np.std(eval_scores))
                    normalized_eval_score_mean = float(np.mean(normalized_eval_scores))
                    normalized_eval_score_std = float(np.std(normalized_eval_scores))

                    eval_fit_log[f"{prefix}/inner_eval_steps"].append(int(fit_step))
                    eval_fit_log[f"{prefix}/inner_score_mean"].append(eval_score_mean)
                    eval_fit_log[f"{prefix}/inner_score_std"].append(eval_score_std)
                    eval_fit_log[f"{prefix}/inner_d4rl_normalized_score_mean"].append(normalized_eval_score_mean)
                    eval_fit_log[f"{prefix}/inner_d4rl_normalized_score_std"].append(normalized_eval_score_std)
                    eval_fit_log[f"{prefix}/final_score_mean"] = eval_score_mean
                    eval_fit_log[f"{prefix}/final_score_std"] = eval_score_std
                    eval_fit_log[f"{prefix}/final_d4rl_normalized_score_mean"] = normalized_eval_score_mean
                    eval_fit_log[f"{prefix}/final_d4rl_normalized_score_std"] = normalized_eval_score_std

                    if normalized_eval_score_mean > best_normalized_score_mean:
                        best_normalized_score_mean = normalized_eval_score_mean
                        eval_fit_log[f"{prefix}/best_score_mean"] = eval_score_mean
                        eval_fit_log[f"{prefix}/best_score_std"] = eval_score_std
                        eval_fit_log[f"{prefix}/best_d4rl_normalized_score_mean"] = normalized_eval_score_mean
                        eval_fit_log[f"{prefix}/best_d4rl_normalized_score_std"] = normalized_eval_score_std

                    print(
                        f"[{prefix}:td3_bc] step {fit_step}/{steps}: "
                        f"loss={step_log['loss']:.4f}, q={step_log['q_mean']:.4f}, "
                        f"lambda={step_log['lambda']:.4f}, bc_loss={step_log['bc_loss']:.4f}, "
                        f"eval_mean={eval_score_mean:.3f}, eval_std={eval_score_std:.3f}, "
                        f"D4RL_mean={normalized_eval_score_mean:.3f}, "
                        f"D4RL_std={normalized_eval_score_std:.3f}"
                    )
        finally:
            for param, requires_grad in zip(self.qf.parameters(), q_requires_grad):
                param.requires_grad_(requires_grad)

        return actor_state, eval_fit_log

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cdaf_state": {
                "qf": self.qf.state_dict(),
                "q_target": self.q_target.state_dict(),
                "q_delayed": self.q_delayed.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "vf": self.vf.state_dict(),
                "v_target": self.v_target.state_dict(),
                "v_delayed": self.v_delayed.state_dict(),
                "v_optimizer": self.v_optimizer.state_dict(),
                "total_it": self.total_it,
            },
            "actor_state": {
                "params": self.actor_state.actor.state_dict(),
                "opt_state": self.actor_state.optimizer.state_dict(),
            },
            "initial_actor_params": self.initial_actor_params,
            "initial_actor_opt_state": self.initial_actor_opt_state,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if "cdaf_state" in state_dict:
            cdaf_state = state_dict["cdaf_state"]
            actor_state = state_dict["actor_state"]

            self.qf.load_state_dict(cdaf_state["qf"])
            self.q_target.load_state_dict(cdaf_state["q_target"])
            self.q_delayed.load_state_dict(cdaf_state["q_delayed"])
            self.q_optimizer.load_state_dict(cdaf_state["q_optimizer"])

            self.vf.load_state_dict(cdaf_state["vf"])
            self.v_target.load_state_dict(cdaf_state["v_target"])
            self.v_delayed.load_state_dict(cdaf_state["v_delayed"])
            self.v_optimizer.load_state_dict(cdaf_state["v_optimizer"])

            self.actor_state.actor.load_state_dict(actor_state["params"])
            self.actor_state.optimizer.load_state_dict(actor_state["opt_state"])
            self.initial_actor_params = copy.deepcopy(
                state_dict.get("initial_actor_params", self.actor_state.actor.state_dict())
            )
            self.initial_actor_opt_state = copy.deepcopy(
                state_dict.get("initial_actor_opt_state", self.actor_state.optimizer.state_dict())
            )
            self.total_it = int(cdaf_state["total_it"])
            return

        # Backward-compatible path for old td3_dw.py checkpoints.
        self.qf.load_state_dict(state_dict["qf"])
        self.q_target.load_state_dict(state_dict["q_target"])
        self.q_delayed.load_state_dict(state_dict["q_delayed"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])

        self.vf.load_state_dict(state_dict["vf"])
        self.v_target.load_state_dict(state_dict["v_target"])
        self.v_delayed.load_state_dict(state_dict["v_delayed"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor_state.actor.load_state_dict(state_dict["actor"])
        self.actor_state.optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.initial_actor_params = copy.deepcopy(
            state_dict.get("actor_reset_state", self.actor_state.actor.state_dict())
        )
        self.initial_actor_opt_state = copy.deepcopy(
            state_dict.get("actor_optimizer_reset_state", self.actor_state.optimizer.state_dict())
        )
        self.total_it = int(state_dict["total_it"])


def save_torch(path: Union[str, Path], obj: Any) -> None:
    torch.save(obj, path)


def load_torch(path: Union[str, Path], device: Union[str, torch.device]) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_checkpoint_path(
    load_model: Union[str, Path],
    run_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[Path, Path]:
    """Return (run_dir, checkpoint_path) for a saved CDAF checkpoint.

    Supported load_model formats:

    1. Direct checkpoint file:
       path/to/checkpoint.pt

    2. Direct run directory:
       path/to/run_dir/
       where path/to/run_dir/checkpoint.pt exists

    3. Parent directory that contains env/seed subdirectory:
       path/to/base_dir/
       where path/to/base_dir/{run_name}/{seed}/checkpoint.pt exists

    Example:
       --load_model logs/tuning/cdaf/0.2/0.1

       resolves to:
       logs/tuning/cdaf/0.2/0.1/CDAF-antmaze-medium-play-v2/0/checkpoint.pt
    """
    load_path = Path(load_model)

    if load_path.is_file():
        return load_path.parent, load_path

    if not load_path.exists():
        raise FileNotFoundError(f"load_model path does not exist: {load_path}")

    candidates: List[Path] = []

    # Case 1: load_model is already the run directory.
    candidates.append(load_path / "checkpoint.pt")

    # Case 2: load_model is a parent directory containing {run_name}/{seed}/checkpoint.pt.
    if run_name is not None and seed is not None:
        candidates.append(load_path / run_name / str(seed) / "checkpoint.pt")

    # Case 3: load_model contains {run_name}/*/checkpoint.pt.
    if run_name is not None:
        run_name_dir = load_path / run_name
        if run_name_dir.exists():
            candidates.extend(sorted(run_name_dir.glob("*/checkpoint.pt")))

    # Case 4: fallback search, but only 2 levels deep to avoid accidentally
    # scanning unrelated large folders.
    candidates.extend(sorted(load_path.glob("*/checkpoint.pt")))
    candidates.extend(sorted(load_path.glob("*/*/checkpoint.pt")))

    # Deduplicate while preserving order.
    seen = set()
    existing_candidates = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            existing_candidates.append(candidate)

    if len(existing_candidates) == 0:
        tried = "\n".join(str(p) for p in candidates[:20])
        raise FileNotFoundError(
            f"checkpoint file not found under: {load_path}\n"
            f"Tried candidates:\n{tried}"
        )

    # Prefer exact {run_name}/{seed}/checkpoint.pt if available.
    if run_name is not None and seed is not None:
        exact = (load_path / run_name / str(seed) / "checkpoint.pt").resolve()
        if exact in existing_candidates:
            return exact.parent, exact

    if len(existing_candidates) > 1:
        found = "\n".join(str(p) for p in existing_candidates)
        raise FileNotFoundError(
            f"Multiple checkpoint.pt files found under {load_path}.\n"
            f"Please provide a more specific --load_model path.\n"
            f"Found:\n{found}"
        )

    checkpoint_path = existing_candidates[0]
    return checkpoint_path.parent, checkpoint_path


@pyrallis.wrap()
def train(config: TrainConfig):
    config = apply_env_hyperparams(config)
    refit_only = config.load_model != "" and int(config.max_timesteps) <= 0
    if not refit_only:
        config = finalize_checkpoint_path(config)

    torch_device = select_torch_device(config.device)
    config.device = str(torch_device)
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

    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=torch_device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None and not refit_only:
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

    print("---------------------------------------")
    print(f"Training {ALGORITHM_NAME}, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    trainer = CDAF(
        max_action=max_action,
        state_dim=state_dim,
        action_dim=action_dim,
        max_steps=config.max_timesteps,
        qf_lr=config.qf_lr,
        vf_lr=config.vf_lr,
        actor_lr=config.actor_lr,
        discount=config.discount,
        tau=config.tau,
        delayed_update_period=config.delayed_update_period,
        min_weight_exponent=config.min_weight_exponent,
        max_weight_exponent=config.max_weight_exponent,
        weight_logit_clip=config.weight_logit_clip,
        beta_min=config.beta_min,
        alpha=config.alpha,
        bc_coef=config.bc_coef,
        seed=seed,
        device=torch_device,
    )

    loaded_run_dir: Optional[Path] = None
    if config.load_model != "":
        loaded_run_dir, checkpoint_path = resolve_checkpoint_path(
            config.load_model,
            run_name=config.name,
            seed=config.seed,
        )
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = load_torch(checkpoint_path, torch_device)
        trainer.load_state_dict(checkpoint)

    if config.log_wandb:
        wandb_init(asdict(config))

    if refit_only:
        if loaded_run_dir is None:
            raise ValueError("refit_only mode requires --load_model")

        actor_refit_dir = loaded_run_dir / config.actor_refit_dir_name
        actor_refit_dir.mkdir(parents=True, exist_ok=True)
        print("---------------------------------------")
        print(f"Actor refit from saved {ALGORITHM_NAME} checkpoint")
        print(f"Saving actor refit outputs to: {actor_refit_dir}")
        print("---------------------------------------")

        fresh_actor_state = trainer._new_initial_actor_state()
        refit_actor_state, refit_log = trainer.fit_actor(
            replay_buffer=replay_buffer,
            actor_state=fresh_actor_state,
            steps=config.refit_actor_steps,
            batch_size=config.eval_actor_batch_size,
            eval_env=env,
            eval_episodes=config.n_episodes,
            eval_seed=config.seed,
            eval_interval=config.eval_actor_eval_freq,
            prefix="actor_refit",
        )

        save_torch(
            actor_refit_dir / "final_actor.pt",
            refit_actor_state.actor.state_dict(),
        )
        save_logs_npz(
            [{"loaded_checkpoint": str(loaded_run_dir / "checkpoint.pt"), **refit_log}],
            str(actor_refit_dir / "fit_eval_logs.npz"),
        )
        with open(actor_refit_dir / "refit_config.yaml", "w") as f:
            pyrallis.dump(config, f)

        if config.log_wandb and wandb.run is not None:
            wandb.save(str(actor_refit_dir / "final_actor.pt"), policy="now")
            wandb.save(str(actor_refit_dir / "fit_eval_logs.npz"), policy="now")
            wandb.save(str(actor_refit_dir / "refit_config.yaml"), policy="now")

        print("---------------------------------------")
        print("Actor refit finished")
        print(f"Saved final actor to: {actor_refit_dir / 'final_actor.pt'}")
        print(f"Saved fit logs to:    {actor_refit_dir / 'fit_eval_logs.npz'}")
        print("---------------------------------------")
        return

    eval_logs: List[Dict[str, Any]] = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        log_dict = trainer.train(batch)

        if config.log_wandb and (t + 1) % config.log_every == 0:
            wandb.log(log_dict, step=trainer.total_it)

        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            if config.reset_actor_on_eval:
                trainer.reset_actor()

            actor_state, eval_fit_log = trainer.fit_actor(
                replay_buffer=replay_buffer,
                actor_state=trainer.actor_state,
                steps=config.eval_actor_steps,
                batch_size=config.eval_actor_batch_size,
                eval_env=env,
                eval_episodes=config.n_episodes,
                eval_seed=config.seed,
                eval_interval=config.eval_actor_eval_freq,
                prefix="fit_actor",
            )
            trainer.actor_state = actor_state

            eval_log: Dict[str, Any] = {
                "timestep": int(t + 1),
                "eval/reward_mean": eval_fit_log["fit_actor/final_score_mean"],
                "eval/reward_std": eval_fit_log["fit_actor/final_score_std"],
                "eval/normalized_score_mean": eval_fit_log["fit_actor/final_d4rl_normalized_score_mean"],
                "eval/normalized_score_std": eval_fit_log["fit_actor/final_d4rl_normalized_score_std"],
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

    if config.checkpoints_path is not None:
        save_torch(
            os.path.join(config.checkpoints_path, "checkpoint.pt"),
            trainer.state_dict(),
        )

        if config.log_wandb and wandb.run is not None:
            wandb.save(os.path.join(config.checkpoints_path, "checkpoint.pt"), policy="now")

        save_and_upload_eval_logs(
            eval_logs=eval_logs,
            checkpoints_path=config.checkpoints_path,
            log_wandb=config.log_wandb,
        )


if __name__ == "__main__":
    train()
