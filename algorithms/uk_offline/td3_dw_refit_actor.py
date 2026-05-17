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
import yaml

TensorBatch = List[torch.Tensor]


@dataclass
class RefitActorConfig:
    # Required I/O
    load_path: str = ""
    save_name: str = "run"
    output_root_name: str = "actor_refit"

    # Device / reproducibility
    device: str = "cuda"
    seed: int = 0

    # Optional overrides. When None, values are loaded from the original config.yaml.
    env: Optional[str] = None
    normalize: Optional[bool] = None
    normalize_reward: Optional[bool] = None
    buffer_size: Optional[int] = None

    # Shared actor fitting params
    actor_method: str = "td3_bc"  # one of: td3_bc, iql
    actor_lr: float = 3e-4
    actor_batch_size: int = 256
    actor_steps: int = 50000
    eval_actor_eval_freq: int = 2000
    n_episodes: int = 10

    # TD3+BC-style actor fitting params
    td3_bc_alpha: float = 2.5
    td3_bc_bc_coef: float = 1.0

    # IQL-style actor fitting params
    iql_beta: float = 3.0
    iql_exp_adv_max: float = 100.0

    # Logging
    log_wandb: bool = True
    project: str = "ORL-BIAS"
    group: str = "ACTOR-REFIT"
    name: str = "ACTOR-REFIT"

    def __post_init__(self):
        if not self.load_path:
            raise ValueError("load_path must be provided")
        if self.actor_method not in {"td3_bc", "iql"}:
            raise ValueError("actor_method must be one of: td3_bc, iql")
        if self.actor_steps <= 0:
            raise ValueError("actor_steps must be > 0")
        if self.actor_batch_size <= 0:
            raise ValueError("actor_batch_size must be > 0")
        if self.eval_actor_eval_freq <= 0:
            raise ValueError("eval_actor_eval_freq must be > 0")
        if self.n_episodes <= 0:
            raise ValueError("n_episodes must be > 0")
        if self.td3_bc_alpha <= 0:
            raise ValueError("td3_bc_alpha must be > 0")
        if self.td3_bc_bc_coef < 0:
            raise ValueError("td3_bc_bc_coef must be >= 0")
        if self.iql_beta <= 0:
            raise ValueError("iql_beta must be > 0")
        if self.iql_exp_adv_max <= 0:
            raise ValueError("iql_exp_adv_max must be > 0")
        if not self.save_name:
            self.save_name = "run"
        self.name = f"{self.name}-{self.actor_method}"

        self.load_path = os.path.join(self.load_path, str(self.seed))

        self.log_wandb = False


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str = "cpu"):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
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


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


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
        dataset["rewards"] -= 1.0


class MLP(nn.Module):
    def __init__(self, dims: List[int], squeeze_output: bool = False):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self.squeeze_output = squeeze_output
        if squeeze_output and dims[-1] != 1:
            raise ValueError("Last dimension must be 1 when squeeze_output=True")

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
        return self(state).cpu().numpy().flatten()


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


@torch.no_grad()
def eval_actor(env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int) -> np.ndarray:
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
    return np.asarray(episode_rewards, dtype=np.float32)


def normalize_episode_scores(env: gym.Env, eval_scores: np.ndarray) -> np.ndarray:
    return np.asarray([env.get_normalized_score(float(score)) * 100.0 for score in eval_scores], dtype=np.float32)


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


def save_fit_eval_logs_npz(history: List[Dict[str, Any]], summary: Dict[str, Any], path: str) -> None:
    data_to_save: Dict[str, np.ndarray] = {}
    if len(history) > 0:
        history_keys = history[0].keys()
        for key in history_keys:
            values = [log[key] for log in history]
            try:
                data_to_save[key] = np.asarray(values)
            except ValueError:
                data_to_save[key] = np.asarray(values, dtype=object)
    for key, value in summary.items():
        data_to_save[key] = np.asarray([value])
    np.savez(path, **data_to_save)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def resolve_checkpoint_paths(load_path: str) -> Tuple[Path, Path, Optional[Path]]:
    p = Path(load_path)
    if p.is_dir():
        checkpoint_path = p / "checkpoint.pt"
        config_path = p / "config.yaml"
        root_dir = p
    else:
        checkpoint_path = p
        root_dir = p.parent
        config_path = root_dir / "config.yaml"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")
    return root_dir, checkpoint_path, config_path if config_path.exists() else None


def load_saved_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def build_output_dir(root_dir: Path, output_root_name: str, save_name: str) -> Path:
    out_dir = root_dir / output_root_name / save_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _update_common_eval_stats(
    summary: Dict[str, Any],
    history: List[Dict[str, Any]],
    actor: Actor,
    env: gym.Env,
    device: str,
    n_episodes: int,
    seed: int,
    fit_step: int,
    extra_log: Dict[str, float],
    best_actor_state: Dict[str, Any],
    best_normalized_mean: float,
    log_wandb: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    eval_scores = eval_actor(env, actor, device=device, n_episodes=n_episodes, seed=seed)
    normalized_eval_scores = normalize_episode_scores(env, eval_scores)

    log = {
        "fit_step": int(fit_step),
        "score_mean": float(np.mean(eval_scores)),
        "score_std": float(np.std(eval_scores)),
        "d4rl_normalized_score_mean": float(np.mean(normalized_eval_scores)),
        "d4rl_normalized_score_std": float(np.std(normalized_eval_scores)),
    }
    log.update(extra_log)
    history.append(log)

    summary["final_score_mean"] = log["score_mean"]
    summary["final_score_std"] = log["score_std"]
    summary["final_d4rl_normalized_score_mean"] = log["d4rl_normalized_score_mean"]
    summary["final_d4rl_normalized_score_std"] = log["d4rl_normalized_score_std"]

    if log["d4rl_normalized_score_mean"] > best_normalized_mean:
        best_normalized_mean = log["d4rl_normalized_score_mean"]
        summary["best_score_mean"] = log["score_mean"]
        summary["best_score_std"] = log["score_std"]
        summary["best_d4rl_normalized_score_mean"] = log["d4rl_normalized_score_mean"]
        summary["best_d4rl_normalized_score_std"] = log["d4rl_normalized_score_std"]
        best_actor_state = copy.deepcopy(actor.state_dict())

    if log_wandb and wandb.run is not None:
        wandb.log({k: to_python_scalar(v) for k, v in log.items() if is_scalar_value(v)}, step=fit_step)

    return summary, best_actor_state, best_normalized_mean


def fit_actor_with_td3bc_loss(
    actor: Actor,
    actor_optimizer: torch.optim.Optimizer,
    qf: QFunction,
    replay_buffer: ReplayBuffer,
    env: gym.Env,
    device: str,
    steps: int,
    batch_size: int,
    eval_interval: int,
    n_episodes: int,
    seed: int,
    td3_bc_alpha: float,
    td3_bc_bc_coef: float,
    log_wandb: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    actor.train()
    qf.eval()

    summary: Dict[str, Any] = {
        "actor_method": "td3_bc",
        "final_loss": np.nan,
        "final_q": np.nan,
        "final_lambda": np.nan,
        "final_bc_loss": np.nan,
        "final_score_mean": np.nan,
        "final_score_std": np.nan,
        "final_d4rl_normalized_score_mean": np.nan,
        "final_d4rl_normalized_score_std": np.nan,
        "best_score_mean": np.nan,
        "best_score_std": np.nan,
        "best_d4rl_normalized_score_mean": np.nan,
        "best_d4rl_normalized_score_std": np.nan,
    }
    history: List[Dict[str, Any]] = []
    best_actor_state: Dict[str, Any] = copy.deepcopy(actor.state_dict())
    best_normalized_mean = -np.inf

    for fit_step in range(1, steps + 1):
        observations, actions, _, _, _ = replay_buffer.sample(batch_size)
        observations = observations.to(device)
        actions = actions.to(device)

        pi = actor(observations)
        q = qf(observations, pi)
        lmbda = td3_bc_alpha / q.abs().mean().detach().clamp(min=1e-6)
        bc_loss = F.mse_loss(pi, actions)
        actor_loss = -lmbda * q.mean() + td3_bc_bc_coef * bc_loss

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        summary["final_loss"] = float(actor_loss.item())
        summary["final_q"] = float(q.mean().item())
        summary["final_lambda"] = float(lmbda.item())
        summary["final_bc_loss"] = float(bc_loss.item())

        if fit_step % eval_interval == 0 or fit_step == steps:
            extra_log = {
                "loss": float(actor_loss.item()),
                "q_mean": float(q.mean().item()),
                "lambda": float(lmbda.item()),
                "bc_loss": float(bc_loss.item()),
                "td3_bc_alpha": float(td3_bc_alpha),
                "td3_bc_bc_coef": float(td3_bc_bc_coef),
            }
            summary, best_actor_state, best_normalized_mean = _update_common_eval_stats(
                summary=summary,
                history=history,
                actor=actor,
                env=env,
                device=device,
                n_episodes=n_episodes,
                seed=seed,
                fit_step=fit_step,
                extra_log=extra_log,
                best_actor_state=best_actor_state,
                best_normalized_mean=best_normalized_mean,
                log_wandb=log_wandb,
            )
            last = history[-1]
            print(
                f"[actor_refit:td3_bc] step {fit_step}/{steps}: "
                f"loss={last['loss']:.4f}, q={last['q_mean']:.4f}, lambda={last['lambda']:.4f}, "
                f"bc_loss={last['bc_loss']:.4f}, bc_coef={last['td3_bc_bc_coef']:.4f}, "
                f"eval_mean={last['score_mean']:.3f}, eval_std={last['score_std']:.3f}, "
                f"D4RL_mean={last['d4rl_normalized_score_mean']:.3f}, "
                f"D4RL_std={last['d4rl_normalized_score_std']:.3f}"
            )

    return summary, history, best_actor_state


def fit_actor_with_iql_loss(
    actor: Actor,
    actor_optimizer: torch.optim.Optimizer,
    qf: QFunction,
    vf: ValueFunction,
    replay_buffer: ReplayBuffer,
    env: gym.Env,
    device: str,
    steps: int,
    batch_size: int,
    eval_interval: int,
    n_episodes: int,
    seed: int,
    iql_beta: float,
    iql_exp_adv_max: float,
    log_wandb: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    actor.train()
    qf.eval()
    vf.eval()

    summary: Dict[str, Any] = {
        "actor_method": "iql",
        "final_loss": np.nan,
        "final_q": np.nan,
        "final_adv_mean": np.nan,
        "final_adv_std": np.nan,
        "final_weight_mean": np.nan,
        "final_bc_loss": np.nan,
        "final_score_mean": np.nan,
        "final_score_std": np.nan,
        "final_d4rl_normalized_score_mean": np.nan,
        "final_d4rl_normalized_score_std": np.nan,
        "best_score_mean": np.nan,
        "best_score_std": np.nan,
        "best_d4rl_normalized_score_mean": np.nan,
        "best_d4rl_normalized_score_std": np.nan,
    }
    history: List[Dict[str, Any]] = []
    best_actor_state: Dict[str, Any] = copy.deepcopy(actor.state_dict())
    best_normalized_mean = -np.inf

    for fit_step in range(1, steps + 1):
        observations, actions, _, _, _ = replay_buffer.sample(batch_size)
        observations = observations.to(device)
        actions = actions.to(device)

        pi = actor(observations)
        q_pi = qf(observations, pi)
        with torch.no_grad():
            q_data = qf(observations, actions)
            v = vf(observations)
            adv = q_data - v
            weights = torch.exp(iql_beta * adv).clamp(max=iql_exp_adv_max)

        bc_losses = ((pi - actions) ** 2).mean(dim=1)
        actor_loss = torch.mean(weights * bc_losses)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        summary["final_loss"] = float(actor_loss.item())
        summary["final_q"] = float(q_pi.mean().item())
        summary["final_adv_mean"] = float(adv.mean().item())
        summary["final_adv_std"] = float(adv.std().item())
        summary["final_weight_mean"] = float(weights.mean().item())
        summary["final_bc_loss"] = float(bc_losses.mean().item())

        if fit_step % eval_interval == 0 or fit_step == steps:
            extra_log = {
                "loss": float(actor_loss.item()),
                "q_mean": float(q_pi.mean().item()),
                "adv_mean": float(adv.mean().item()),
                "adv_std": float(adv.std().item()),
                "weight_mean": float(weights.mean().item()),
                "bc_loss": float(bc_losses.mean().item()),
                "iql_beta": float(iql_beta),
                "iql_exp_adv_max": float(iql_exp_adv_max),
            }
            summary, best_actor_state, best_normalized_mean = _update_common_eval_stats(
                summary=summary,
                history=history,
                actor=actor,
                env=env,
                device=device,
                n_episodes=n_episodes,
                seed=seed,
                fit_step=fit_step,
                extra_log=extra_log,
                best_actor_state=best_actor_state,
                best_normalized_mean=best_normalized_mean,
                log_wandb=log_wandb,
            )
            last = history[-1]
            print(
                f"[actor_refit:iql] step {fit_step}/{steps}: "
                f"loss={last['loss']:.4f}, q={last['q_mean']:.4f}, adv_mean={last['adv_mean']:.4f}, "
                f"adv_std={last['adv_std']:.4f}, weight_mean={last['weight_mean']:.4f}, "
                f"bc_loss={last['bc_loss']:.4f}, eval_mean={last['score_mean']:.3f}, "
                f"eval_std={last['score_std']:.3f}, D4RL_mean={last['d4rl_normalized_score_mean']:.3f}, "
                f"D4RL_std={last['d4rl_normalized_score_std']:.3f}"
            )

    return summary, history, best_actor_state


@pyrallis.wrap()
def main(config: RefitActorConfig):
    root_dir, checkpoint_path, config_path = resolve_checkpoint_paths(config.load_path)
    saved_cfg = load_saved_config(config_path)

    env_name = config.env or saved_cfg.get("env")
    if env_name is None:
        raise ValueError("env must be provided either in the refit config or the original config.yaml")
    if "antmaze" in env_name or "maze2d" in env_name:
        config.n_episodes = 100
    print(config.n_episodes)
    config.env = env_name
    normalize = config.normalize if config.normalize is not None else bool(saved_cfg.get("normalize", True))
    normalize_reward = (
        config.normalize_reward if config.normalize_reward is not None else bool(saved_cfg.get("normalize_reward", False))
    )
    buffer_size = config.buffer_size if config.buffer_size is not None else int(saved_cfg.get("buffer_size", 2_000_000))

    output_dir = build_output_dir(root_dir, f"{config.actor_method}_{config.output_root_name}", config.save_name)
    print(f"Original checkpoint: {checkpoint_path}")
    print(f"Original config: {config_path if config_path is not None else 'not found'}")
    print(f"Output dir: {output_dir}")

    env = gym.make(env_name)
    set_seed(config.seed, env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    dataset = d4rl.qlearning_dataset(env)
    if normalize_reward:
        modify_reward(dataset, env_name)

    if normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size, config.device)
    replay_buffer.load_d4rl_dataset(dataset)

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    qf = QFunction(state_dim, action_dim).to(config.device)
    vf = ValueFunction(state_dim).to(config.device)

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    if "qf" not in checkpoint or "vf" not in checkpoint:
        raise KeyError("checkpoint must contain 'qf' and 'vf' state dicts")
    qf.load_state_dict(checkpoint["qf"])
    vf.load_state_dict(checkpoint["vf"])
    qf.requires_grad_(False)
    vf.requires_grad_(False)
    qf.eval()
    vf.eval()

    with open(output_dir / "refit_config.yaml", "w") as f:
        pyrallis.dump(config, f)

    if config.log_wandb:
        wandb_init(asdict(config))

    if config.actor_method == "td3_bc":
        summary, history, best_actor_state = fit_actor_with_td3bc_loss(
            actor=actor,
            actor_optimizer=actor_optimizer,
            qf=qf,
            replay_buffer=replay_buffer,
            env=env,
            device=config.device,
            steps=config.actor_steps,
            batch_size=config.actor_batch_size,
            eval_interval=config.eval_actor_eval_freq,
            n_episodes=config.n_episodes,
            seed=config.seed,
            td3_bc_alpha=config.td3_bc_alpha,
            td3_bc_bc_coef=config.td3_bc_bc_coef,
            log_wandb=config.log_wandb,
        )
    else:
        summary, history, best_actor_state = fit_actor_with_iql_loss(
            actor=actor,
            actor_optimizer=actor_optimizer,
            qf=qf,
            vf=vf,
            replay_buffer=replay_buffer,
            env=env,
            device=config.device,
            steps=config.actor_steps,
            batch_size=config.actor_batch_size,
            eval_interval=config.eval_actor_eval_freq,
            n_episodes=config.n_episodes,
            seed=config.seed,
            iql_beta=config.iql_beta,
            iql_exp_adv_max=config.iql_exp_adv_max,
            log_wandb=config.log_wandb,
        )

    torch.save(actor.state_dict(), output_dir / "final_actor.pt")
    torch.save(best_actor_state, output_dir / "best_actor.pt")
    save_fit_eval_logs_npz(history, summary, str(output_dir / "fit_eval_logs.npz"))

    if config.log_wandb and wandb.run is not None:
        wandb.save(str(output_dir / "final_actor.pt"), policy="now")
        wandb.save(str(output_dir / "best_actor.pt"), policy="now")
        wandb.save(str(output_dir / "fit_eval_logs.npz"), policy="now")
        wandb.log({f"summary/{k}": to_python_scalar(v) for k, v in summary.items() if is_scalar_value(v)})

    print("---------------------------------------")
    print("Actor refit finished")
    print(f"Method: {config.actor_method}")
    print(f"Final D4RL mean: {summary['final_d4rl_normalized_score_mean']:.3f}")
    print(f"Best  D4RL mean: {summary['best_d4rl_normalized_score_mean']:.3f}")
    print(f"Saved final actor to: {output_dir / 'final_actor.pt'}")
    print(f"Saved best actor to:  {output_dir / 'best_actor.pt'}")
    print("---------------------------------------")


if __name__ == "__main__":
    main()
