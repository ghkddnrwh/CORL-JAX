import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import collections
import pickle
from functools import partial
import random
import re
import sys
import time
import uuid
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

try:
    import scipy.linalg as scipy_linalg
    if not hasattr(scipy_linalg, "tril"):
        scipy_linalg.tril = np.tril
    if not hasattr(scipy_linalg, "triu"):
        scipy_linalg.triu = np.triu
except ImportError:
    pass

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pyrallis
from flax import serialization, struct
from flax.core import FrozenDict, freeze, unfreeze
from flax.core.frozen_dict import FrozenDict as DatasetFrozenDict
from gymnasium.spaces import Box
from tqdm.auto import trange

try:
    import gymnasium
except ImportError:
    gymnasium = None

try:
    import ogbench
except ImportError:
    ogbench = None

try:
    import wandb
except ImportError:
    class _UnavailableWandb:
        run = None

        def init(self, *args, **kwargs):
            raise ImportError(
                "wandb is unavailable in this environment; run with --log_wandb False "
                "or install wandb."
            )

        def save(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

        def mark_preempting(self, *args, **kwargs):
            return None

    wandb = _UnavailableWandb()


d4rl = None

# Automatic training resume utilities shared by all offline-RL algorithms.
_PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "algorithms").is_dir()
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.uk_offline.common.checkpointing import (
    DEFAULT_IDENTITY_IGNORED_FIELDS,
    TrainingCheckpointManager,
    best_eval_metric,
    evaluation_is_due,
    find_eval_log,
    upsert_eval_log,
)

ALGORITHM_NAME = "FQL"
ALGORITHM_FULL_NAME = "Flow Q-Learning with IQL_JAX-style experiment plumbing"


# -----------------------------------------------------------------------------
# Config / experiment plumbing
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Experiment.
    device: str = "gpu"
    env: str = "cube-double-play-singletask-v0"
    seed: int = 0
    eval_seed: int = 42

    # Schedule.
    max_timesteps: int = int(1e6)
    eval_freq: int = int(100000)
    n_episodes: int = 50

    # I/O and modes.
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    mode: str = "train"  # FQL wrapper currently supports train only; load_model resumes/evaluates initialization.
    hyperparams_path: Optional[str] = "hyperparams/fql_jax.yml"
    use_hyperparams: bool = True
    dataset_name: Optional[str] = None

    # Dataset / env options from FQL main.py.
    batch_size: int = 256
    normalize_states: bool = False  # Kept for interface compatibility; FQL original does not use it.
    normalize_reward: bool = False  # Kept for interface compatibility; FQL original does not use it.
    action_clip_eps: Optional[float] = 1e-5
    frame_stack: Optional[int] = None
    p_aug: Optional[float] = None

    # FQL algorithm parameters.
    lr: Optional[float] = None
    actor_learning_rate: float = 3e-4  # Alias used by the user's runner; FQL uses one shared optimizer.
    critic_learning_rate: float = 3e-4  # Accepted but ignored unless lr is unset.
    hidden_dim: int = 512
    actor_n_hiddens: int = 4
    critic_n_hiddens: int = 4
    actor_hidden_dims: Optional[Tuple[int, ...]] = None
    value_hidden_dims: Optional[Tuple[int, ...]] = None
    layer_norm: Optional[bool] = None
    actor_layer_norm: Optional[bool] = None
    critic_ln: bool = True       # Alias for FQL layer_norm.
    actor_ln: bool = False       # Alias for FQL actor_layer_norm.
    discount: float = 0.99
    tau: float = 0.005
    q_agg: str = "mean"
    alpha: float = 10.0
    flow_steps: int = 10
    normalize_q_loss: bool = False
    encoder: Optional[str] = None

    # Compatibility fields that may appear in shared YAMLs. They are ignored by FQL.
    num_critics: int = 2
    policy_noise: float = 0.0
    noise_clip: float = 0.0
    policy_freq: int = 1
    normalize_q: bool = False
    tanh_squash: bool = True
    actor_fc_scale: float = 0.01
    actor_refit_dir_name: str = "actor_refit"

    # Logging / saving.
    project: str = "ORL-SMOOTH"
    group: str = "FQL-JAX"
    name: str = "FQL-JAX"
    log_wandb: bool = True
    log_every: int = 5000
    save_best_model: bool = True
    eval_at_first_step: bool = True

    checkpoint_freq: int = int(25e3)
    save_final_model: bool = True
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        normalize_config_aliases(self)
        refresh_algorithm_names(self)
        validate_config(self)


def normalize_config_aliases(config: TrainConfig) -> None:
    if config.dataset_name is not None:
        config.env = config.dataset_name

    if config.mode == "refit":
        config.mode = "actor_refit"

    if config.actor_hidden_dims is not None and not isinstance(config.actor_hidden_dims, tuple):
        config.actor_hidden_dims = tuple(config.actor_hidden_dims)
    if config.value_hidden_dims is not None and not isinstance(config.value_hidden_dims, tuple):
        config.value_hidden_dims = tuple(config.value_hidden_dims)

    if config.actor_hidden_dims is None:
        config.actor_hidden_dims = tuple([int(config.hidden_dim)] * int(config.actor_n_hiddens))
    if config.value_hidden_dims is None:
        config.value_hidden_dims = tuple([int(config.hidden_dim)] * int(config.critic_n_hiddens))

    if config.layer_norm is None:
        config.layer_norm = bool(config.critic_ln)
    if config.actor_layer_norm is None:
        config.actor_layer_norm = bool(config.actor_ln)

    if config.lr is None:
        config.lr = float(config.actor_learning_rate)


def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = "ORL-SMOOTH"
    config.group = f"{ALGORITHM_NAME}-JAX"
    config.name = f"{ALGORITHM_NAME}-JAX-{config.env}"


def validate_config(config: TrainConfig) -> None:
    assert config.mode in ("train", "actor_refit"), "mode must be train or actor_refit"
    if config.mode == "actor_refit":
        raise NotImplementedError("FQL actor_refit is not implemented. Use mode='train'.")
    assert config.batch_size > 0
    assert config.eval_freq > 0
    assert config.n_episodes > 0
    assert config.max_timesteps >= 0
    assert config.log_every > 0
    assert config.checkpoint_freq > 0
    assert config.discount >= 0.0 and config.discount <= 1.0
    assert config.tau >= 0.0 and config.tau <= 1.0
    assert config.q_agg in ("mean", "min")
    assert config.alpha >= 0.0
    assert config.flow_steps > 0
    assert config.lr is not None and config.lr > 0.0
    if abs(float(config.actor_learning_rate) - float(config.critic_learning_rate)) > 0.0:
        print(
            "Warning: FQL uses one shared optimizer for all modules. "
            f"Using lr={config.lr}; critic_learning_rate={config.critic_learning_rate} is ignored."
        )
    if config.normalize_states:
        print("Warning: normalize_states=True changes the original FQL codepath; this wrapper ignores it.")
    if config.normalize_reward:
        print("Warning: normalize_reward=True is not used by the original FQL codepath; this wrapper ignores it.")


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
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return tuple(value)
    return value


def apply_env_hyperparams(config: TrainConfig) -> TrainConfig:
    normalize_config_aliases(config)
    if not config.use_hyperparams or config.hyperparams_path is None:
        refresh_algorithm_names(config)
        validate_config(config)
        return config

    hparam_path = Path(config.hyperparams_path)
    if not hparam_path.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {hparam_path}.")

    with open(hparam_path, "r") as f:
        all_hyperparams = yaml.safe_load(f) or {}

    if config.env not in all_hyperparams:
        print(
            f"No hyperparameters found for env '{config.env}' in {hparam_path}. "
            "Using dataclass/CLI values."
        )
        refresh_algorithm_names(config)
        validate_config(config)
        return config

    env_hyperparams = all_hyperparams[config.env] or {}
    cli_overrides = _cli_overridden_fields()
    aliases = {
        "dataset_name": "env",
        "env_name": "env",
        "offline_steps": "max_timesteps",
        "n_timesteps": "max_timesteps",
        "eval_interval": "eval_freq",
        "eval_episodes": "n_episodes",
        "learning_rate": "lr",
    }
    config_fields = set(TrainConfig.__dataclass_fields__.keys())
    applied, skipped_unknown, skipped_cli = [], [], []
    applied_fields = set()

    for raw_key, raw_value in env_hyperparams.items():
        key = aliases.get(raw_key, raw_key)
        if key not in config_fields:
            skipped_unknown.append(raw_key)
            continue
        if key in applied_fields:
            continue
        if key in cli_overrides or raw_key in cli_overrides:
            skipped_cli.append(raw_key)
            continue
        setattr(config, key, _coerce_hparam_value(raw_value))
        applied.append(f"{raw_key}->{key}" if raw_key != key else key)
        applied_fields.add(key)

    normalize_config_aliases(config)
    refresh_algorithm_names(config)
    validate_config(config)

    if applied:
        print(f"Loaded hyperparameters for {config.env} from {hparam_path}: {', '.join(applied)}")
    if skipped_cli:
        print(f"Kept CLI overrides for: {', '.join(skipped_cli)}")
    if skipped_unknown:
        print(f"Ignored unknown hyperparameter keys for FQL: {', '.join(skipped_unknown)}")
    return config


def finalize_checkpoint_path(config: TrainConfig) -> TrainConfig:
    if config.checkpoints_path is not None:
        config.checkpoints_path = os.path.join(config.checkpoints_path, config.name, str(config.seed))
    return config


def select_jax_device(device: str):
    backend = device.lower()
    if backend == "cuda":
        backend = "gpu"
    try:
        dev = jax.devices(backend)[0]
    except Exception:
        print(f"Requested JAX backend '{device}' is not available. Falling back to default device.")
        dev = jax.devices()[0]
    print(f"Using JAX device: {dev}")
    return dev


def tree_to_device(tree, device):
    return jax.device_put(tree, device)


def set_seed(seed: int, env: Optional[Any] = None):
    if env is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            try:
                env.seed(seed)
            except Exception:
                pass
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def wandb_init(config: dict) -> None:
    run = wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    if run is not None and hasattr(run, "log_code"):
        run.log_code(".")


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
    if isinstance(value, jnp.ndarray) and value.ndim == 0:
        return float(value)
    return value


def save_logs_npz(logs: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    if len(logs) == 0:
        return
    keys = logs[0].keys()
    data_to_save: Dict[str, np.ndarray] = {}
    for key in keys:
        values = [log.get(key, np.nan) for log in logs]
        try:
            data_to_save[key] = np.asarray(values)
        except ValueError:
            data_to_save[key] = np.asarray(values, dtype=object)
    np.savez(path, **data_to_save)


def save_and_upload_eval_logs(eval_logs, checkpoints_path, log_wandb):
    if checkpoints_path is None or len(eval_logs) == 0:
        return
    eval_logs_path = os.path.join(checkpoints_path, "eval_logs.npz")
    save_logs_npz(eval_logs, eval_logs_path)
    if log_wandb and wandb.run is not None:
        wandb.save(eval_logs_path, policy="now")


def save_pickle(path: Union[str, Path], obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Original FQL dataset/env/evaluation utilities, adapted only for a single file
# -----------------------------------------------------------------------------

def get_size(data):
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=("padding",))
def random_crop(img, crop_from, padding):
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(imgs, crop_froms, padding):
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(DatasetFrozenDict):
    @classmethod
    def create(cls, freeze_arrays=True, **fields):
        data = fields
        assert "observations" in data
        if freeze_arrays:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None
        self.p_aug = None
        self.return_next_actions = False
        self.terminal_locs = np.nonzero(self["terminals"] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side="right") - 1]
            obs = []
            next_obs = []
            for i in reversed(range(self.frame_stack)):
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self["observations"]))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self["observations"]))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self["next_observations"]))
            batch["observations"] = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs, axis=-1), *obs)
            batch["next_observations"] = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs, axis=-1), *next_obs)
        if self.p_aug is not None:
            if np.random.rand() < self.p_aug:
                self.augment(batch, ["observations", "next_observations"])
        return batch

    def get_subset(self, idxs):
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            result["next_actions"] = self._dict["actions"][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    @classmethod
    def create(cls, transition, size):
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)


class EpisodeMonitor(gymnasium.Wrapper):
    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}
        if terminated or truncated:
            info["episode"] = {}
            info["episode"]["final_reward"] = reward
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time
            if hasattr(self.unwrapped, "get_normalized_score"):
                info["episode"]["normalized_return"] = self.unwrapped.get_normalized_score(info["episode"]["return"]) * 100.0
        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)
        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if "goal" in info:
            info["goal"] = np.concatenate([info["goal"]] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def d4rl_make_env(env_name):
    import gymnasium as gymnasium_module
    env = gymnasium_module.make("GymV21Environment-v0", env_id=env_name)
    return EpisodeMonitor(env)


def d4rl_get_dataset(env, env_name):
    global d4rl
    if d4rl is None:
        import d4rl as d4rl_module
        d4rl = d4rl_module
    dataset = d4rl.qlearning_dataset(env)
    terminals = np.zeros_like(dataset["rewards"])
    masks = np.zeros_like(dataset["rewards"])
    rewards = dataset["rewards"].copy().astype(np.float32)
    if "antmaze" in env_name:
        for i in range(len(terminals) - 1):
            terminals[i] = float(np.linalg.norm(dataset["observations"][i + 1] - dataset["next_observations"][i]) > 1e-6)
            masks[i] = 1 - dataset["terminals"][i]
        rewards = rewards - 1.0
    else:
        for i in range(len(terminals) - 1):
            if np.linalg.norm(dataset["observations"][i + 1] - dataset["next_observations"][i]) > 1e-6 or dataset["terminals"][i] == 1.0:
                terminals[i] = 1
            else:
                terminals[i] = 0
            masks[i] = 1 - dataset["terminals"][i]
    masks[-1] = 1 - dataset["terminals"][-1]
    terminals[-1] = 1
    return Dataset.create(
        observations=dataset["observations"].astype(np.float32),
        actions=dataset["actions"].astype(np.float32),
        next_observations=dataset["next_observations"].astype(np.float32),
        terminals=terminals.astype(np.float32),
        rewards=rewards,
        masks=masks,
    )


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5):
    if "singletask" in env_name:
        if ogbench is None:
            raise ImportError("OGBench environment requested, but ogbench is not installed.")
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=[".*privileged.*", ".*proprio.*"])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=[".*privileged.*", ".*proprio.*"])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    elif "antmaze" in env_name and ("diverse" in env_name or "play" in env_name or "umaze" in env_name):
        env = d4rl_make_env(env_name)
        eval_env = d4rl_make_env(env_name)
        train_dataset = d4rl_get_dataset(env, env_name)
        val_dataset = None
    elif "pen" in env_name or "hammer" in env_name or "relocate" in env_name or "door" in env_name:
        env = d4rl_make_env(env_name)
        eval_env = d4rl_make_env(env_name)
        train_dataset = d4rl_get_dataset(env, env_name)
        val_dataset = None
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset["actions"], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset["actions"], -1 + action_clip_eps, 1 - action_clip_eps))
            )
    return env, eval_env, train_dataset, val_dataset


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)
    return wrapped


def evaluate(agent, env, num_eval_episodes=50, eval_temperature=0, seed=0):
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(seed))
    stats = collections.defaultdict(list)
    returns = []
    successes = []
    for _ in trange(num_eval_episodes, desc="Eval", leave=False):
        observation, info = env.reset()
        done = False
        while not done:
            action = actor_fn(observations=observation, temperature=eval_temperature)
            action = np.array(action)
            action = np.clip(action, -1, 1)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation = next_observation
        flat_info = flatten(info)
        add_to(stats, flat_info)
        if "episode.return" in flat_info:
            returns.append(flat_info["episode.return"])
        if "success" in flat_info:
            successes.append(flat_info["success"])
        elif "episode.success" in flat_info:
            successes.append(flat_info["episode.success"])
    averaged = {k: float(np.mean(v)) for k, v in stats.items()}
    return_array = np.asarray(returns, dtype=np.float32)
    success_array = np.asarray(successes, dtype=np.float32)
    return averaged, return_array, success_array


# -----------------------------------------------------------------------------
# Original FQL network / TrainState / agent, adapted only for standalone use
# -----------------------------------------------------------------------------

def nonpytree_field():
    return flax.struct.field(pytree_node=False)


class ModuleDict(nn.Module):
    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f"When name is not specified, kwargs must contain arguments for each module. "
                    f"Got {kwargs.keys()} but module keys are {self.modules.keys()}"
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out
        return self.modules[name](*args, **kwargs)


class FQLTrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Any = flax.struct.field(pytree_node=False)
    model_def: Any = flax.struct.field(pytree_node=False)
    params: Any
    tx: Any = flax.struct.field(pytree_node=False)
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        opt_state = tx.init(params) if tx is not None else None
        return cls(step=1, apply_fn=model_def.apply, model_def=model_def, params=params, tx=tx, opt_state=opt_state, **kwargs)

    def __call__(self, *args, params=None, method=None, **kwargs):
        if params is None:
            params = self.params
        variables = {"params": params}
        method_name = getattr(self.model_def, method) if method is not None else None
        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        return partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)

    def apply_loss_fn(self, loss_fn):
        # Original FQL also logs gradient statistics, but those stats make JIT compilation heavy.
        # Removing them does not change the optimization update.
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
        return self.apply_gradients(grads=grads), info


def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class Identity(nn.Module):
    def __call__(self, x):
        return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class TransformedWithMode(distrax.Transformed):
    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Value(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: Optional[nn.Module] = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        self.value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions=None):
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        return self.value_net(inputs).squeeze(-1)


class ActorVectorField(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: Optional[nn.Module] = None

    def setup(self):
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)
        return self.mlp(inputs)


class ResnetStack(nn.Module):
    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding="SAME",
        )(x)
        if self.max_pooling:
            conv_out = nn.max_pool(conv_out, window_shape=(3, 3), padding="SAME", strides=(2, 2))
        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(features=self.num_features, kernel_size=(3, 3), strides=1, padding="SAME", kernel_init=initializer)(conv_out)
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(features=self.num_features, kernel_size=(3, 3), strides=1, padding="SAME", kernel_init=initializer)(conv_out)
            conv_out += block_input
        return conv_out


class ImpalaEncoder(nn.Module):
    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: Optional[float] = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        self.stack_blocks = [
            ResnetStack(num_features=self.stack_sizes[i] * self.width, num_blocks=self.num_blocks)
            for i in range(len(self.stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0
        conv_out = x
        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)
        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))
        return MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)


encoder_modules = {
    "impala": ImpalaEncoder,
    "impala_debug": partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    "impala_small": partial(ImpalaEncoder, num_blocks=1),
    "impala_large": partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
}


class FQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = flax.struct.field(pytree_node=False)

    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch["next_observations"], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)
        next_qs = self.network.select("target_critic")(batch["next_observations"], actions=next_actions)
        next_q = next_qs.min(axis=0) if self.config["q_agg"] == "min" else next_qs.mean(axis=0)
        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q
        q = self.network.select("critic")(batch["observations"], actions=batch["actions"], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch["actions"].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch["actions"]
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_bc_flow")(batch["observations"], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch["observations"], noises=noises)
        actor_actions = self.network.select("actor_onestep_flow")(batch["observations"], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select("critic")(batch["observations"], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config["normalize_q_loss"]:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.config["alpha"] * distill_loss + q_loss
        actions = self.sample_actions(batch["observations"], seed=rng)
        mse = jnp.mean((actions - batch["actions"]) ** 2)
        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "q": q.mean(),
            "mse": mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v
        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        params = unfreeze(network.params)
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            params[f"modules_{module_name}"],
            params[f"modules_target_{module_name}"],
        )
        params[f"modules_target_{module_name}"] = new_target_params
        return network.replace(params=freeze(params))

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self.target_update(new_network, "critic")
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        action_seed, _ = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (*observations.shape[: -len(self.config["ob_dims"])], self.config["action_dim"]),
        )
        actions = self.network.select("actor_onestep_flow")(observations, noises)
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        if self.config["encoder"] is not None:
            observations = self.network.select("actor_bc_flow_encoder")(observations)
        actions = noises
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            vels = self.network.select("actor_bc_flow")(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config["flow_steps"]
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = {}
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor_bc_flow"] = encoder_module()
            encoders["actor_onestep_flow"] = encoder_module()

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
            encoder=encoders.get("critic"),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_bc_flow"),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_onestep_flow"),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        if encoders.get("actor_bc_flow") is not None:
            network_info["actor_bc_flow_encoder"] = (encoders.get("actor_bc_flow"), (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        variables = network_def.init(init_rng, **network_args)
        params = unfreeze(variables["params"])
        params["modules_target_critic"] = copy.deepcopy(params["modules_critic"])
        network = FQLTrainState.create(network_def, freeze(params), tx=network_tx)

        cfg = dict(config)
        cfg["ob_dims"] = ob_dims
        cfg["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**cfg))


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------

def config_to_agent_config(config: TrainConfig) -> Dict[str, Any]:
    return dict(
        agent_name="fql",
        lr=float(config.lr),
        batch_size=int(config.batch_size),
        actor_hidden_dims=tuple(config.actor_hidden_dims),
        value_hidden_dims=tuple(config.value_hidden_dims),
        layer_norm=bool(config.layer_norm),
        actor_layer_norm=bool(config.actor_layer_norm),
        discount=float(config.discount),
        tau=float(config.tau),
        q_agg=str(config.q_agg),
        alpha=float(config.alpha),
        flow_steps=int(config.flow_steps),
        normalize_q_loss=bool(config.normalize_q_loss),
        encoder=config.encoder,
    )


def make_checkpoint_payload(agent: FQLAgent, config: TrainConfig) -> Dict[str, Any]:
    return {
        "agent": serialization.to_state_dict(agent),
        "config": asdict(config),
    }


def save_checkpoint(checkpoint_path, agent: FQLAgent, config: TrainConfig, log_wandb: bool) -> None:
    save_pickle(checkpoint_path, make_checkpoint_payload(agent=agent, config=config))
    if log_wandb and wandb.run is not None:
        wandb.save(str(checkpoint_path), policy="now")


def load_checkpoint(agent: FQLAgent, checkpoint_path: Union[str, Path]) -> FQLAgent:
    checkpoint = load_pickle(checkpoint_path)
    state = checkpoint["agent"] if isinstance(checkpoint, dict) and "agent" in checkpoint else checkpoint
    return serialization.from_state_dict(agent, state)


def resolve_checkpoint_path(load_model: Union[str, Path], run_name: Optional[str] = None, seed: Optional[int] = None) -> Tuple[Path, Path]:
    load_path = Path(load_model)
    if load_path.is_file():
        return load_path.parent, load_path
    if not load_path.exists():
        raise FileNotFoundError(f"load_model path does not exist: {load_path}")
    for filename in ("checkpoint.pkl", "best_checkpoint.pkl"):
        direct = load_path / filename
        if direct.exists():
            return direct.parent, direct
    if run_name is not None and seed is not None:
        for filename in ("checkpoint.pkl", "best_checkpoint.pkl"):
            exact = load_path / run_name / str(seed) / filename
            if exact.exists():
                return exact.parent, exact
    candidates = []
    candidates.extend(sorted(load_path.glob("*/checkpoint.pkl")))
    candidates.extend(sorted(load_path.glob("*/best_checkpoint.pkl")))
    candidates.extend(sorted(load_path.glob("*/*/checkpoint.pkl")))
    candidates.extend(sorted(load_path.glob("*/*/best_checkpoint.pkl")))
    candidates = [c.resolve() for c in candidates if c.exists()]
    if len(candidates) == 0:
        raise FileNotFoundError(f"checkpoint file not found under: {load_path}")
    if len(candidates) > 1:
        found = "\n".join(str(c) for c in candidates)
        raise FileNotFoundError(f"Multiple checkpoint files found under {load_path}. Please provide a specific path.\n{found}")
    return candidates[0].parent, candidates[0]


# -----------------------------------------------------------------------------
# Main train entry
# -----------------------------------------------------------------------------

def _train_impl(config: TrainConfig):
    normalize_config_aliases(config)
    config = apply_env_hyperparams(config)
    config = finalize_checkpoint_path(config)

    checkpoint_manager = None
    checkpoint_preparation = None
    if config.checkpoints_path is not None and (config.mode == "train"):
        current_config_dict = asdict(config)
        checkpoint_manager = TrainingCheckpointManager(
            run_dir=config.checkpoints_path,
            current_config=current_config_dict,
            default_config=asdict(TrainConfig()),
            max_timesteps=int(config.max_timesteps),
            checkpoint_type="fql_jax_training_progress",
            identity_ignored_fields=DEFAULT_IDENTITY_IGNORED_FIELDS,
            checkpoint_version=2,
            accepted_checkpoint_versions=(1, 2),
            wandb_enabled=bool(config.log_wandb),
            wandb_entity=getattr(config, "wandb_entity", None),
            wandb_project=config.project,
            final_checkpoint_name="checkpoint.pkl",
        )
        checkpoint_preparation = checkpoint_manager.prepare()
        print(checkpoint_preparation.message)
        if checkpoint_preparation.is_completed:
            return

    jax_device = select_jax_device(config.device)
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        config.env,
        frame_stack=config.frame_stack,
        action_clip_eps=config.action_clip_eps,
    )

    # Match FQL main.py dataset setup. For pure offline training without online fine-tuning,
    # the replay buffer is just a copy of the offline dataset with extra capacity.
    train_dataset = Dataset.create(**train_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset),
        size=max(int(2000000), train_dataset.size + 1),
    )
    replay_buffer = train_dataset
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = config.p_aug
            dataset.frame_stack = config.frame_stack
            dataset.return_next_actions = False

    set_seed(config.seed, eval_env)


    print("---------------------------------------")
    print(f"Training {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")

    random.seed(config.seed)
    np.random.seed(config.seed)
    example_batch = replay_buffer.sample(1)
    agent_config = config_to_agent_config(config)
    agent = FQLAgent.create(
        config.seed,
        example_batch["observations"],
        example_batch["actions"],
        agent_config,
    )
    agent = tree_to_device(agent, jax_device)

    if config.load_model != "":
        _, checkpoint_path = resolve_checkpoint_path(config.load_model, run_name=config.name, seed=config.seed)
        print(f"Loading checkpoint from: {checkpoint_path}")
        agent = load_checkpoint(agent, checkpoint_path)
        agent = tree_to_device(agent, jax_device)

    training_timestep = 0

    def _progress_state():
        payload = make_checkpoint_payload(agent=agent, config=config)
        payload["_training_timestep"] = int(training_timestep)
        return payload

    def _final_state():
        return make_checkpoint_payload(agent=agent, config=config)

    def _load_progress_state(payload):
        nonlocal agent, training_timestep
        raw_agent = payload["agent"] if isinstance(payload, dict) and "agent" in payload else payload
        agent = serialization.from_state_dict(agent, raw_agent)
        agent = tree_to_device(agent, jax_device)
        training_timestep = int(payload.get("_training_timestep", 0)) if isinstance(payload, dict) else 0

    def _training_timestep():
        return int(training_timestep)

    start_timestep = _training_timestep()
    eval_logs: List[Dict[str, Any]] = []
    if checkpoint_manager is not None:
        if checkpoint_preparation.is_resuming:
            start_timestep, eval_logs, _ = checkpoint_manager.restore(
                load_trainer_state=_load_progress_state,
                get_restored_timestep=_training_timestep,
            )
            print(f"Restored training at timestep {start_timestep}.")
        else:
            start_timestep = _training_timestep()
            checkpoint_manager.save_progress(
                timestep=start_timestep,
                trainer_state=_progress_state(),
                eval_logs=eval_logs,
                status="running",
            )

    if config.log_wandb:
        if checkpoint_manager is not None:
            try:
                checkpoint_manager.initialize_wandb(
                    wandb_module=wandb,
                    config=asdict(config),
                    code_root=_PROJECT_ROOT,
                )
            except Exception as exc:
                print(f"Warning: W&B resume failed: {exc}")
                print("Continuing local training with a new W&B run.")
                if getattr(wandb, "run", None) is not None:
                    wandb.finish(exit_code=1)
                checkpoint_manager.initialize_fresh_wandb(
                    wandb_module=wandb,
                    config=asdict(config),
                    code_root=_PROJECT_ROOT,
                )
        else:
            wandb_init(asdict(config))


    def _wandb_log(metrics, step):
        if not config.log_wandb:
            return
        if checkpoint_manager is not None:
            checkpoint_manager.log_wandb(metrics, int(step))
        else:
            wandb.log(metrics, step=int(step))


    best_eval_metric_mean = best_eval_metric(eval_logs)
    last_update_info: Dict[str, Any] = {}
    first_time = time.time()
    last_time = time.time()

    def _evaluation_required(step):
        step = int(step)
        return (
            (config.eval_at_first_step and step == 1)
            or step % int(config.eval_freq) == 0
            or step == int(config.max_timesteps)
        )

    try:
        for i in trange(start_timestep + 1, int(config.max_timesteps) + 1, desc="FQL Training"):
            batch = replay_buffer.sample(int(config.batch_size))
            agent, update_info = agent.update(batch)
            training_timestep = int(i)
            last_update_info = {k: float(jax.device_get(v)) for k, v in update_info.items()}
    
            if config.log_wandb and i % int(config.log_every) == 0:
                train_metrics = {f"training/{k}": v for k, v in last_update_info.items()}
                if val_dataset is not None:
                    val_batch = val_dataset.sample(int(config.batch_size))
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                    val_info = {k: float(jax.device_get(v)) for k, v in val_info.items()}
                    train_metrics.update({f"validation/{k}": v for k, v in val_info.items()})
                train_metrics["time/epoch_time"] = (time.time() - last_time) / int(config.log_every)
                train_metrics["time/total_time"] = time.time() - first_time
                last_time = time.time()
                _wandb_log(train_metrics, i)
    
            should_eval = (
                (config.eval_at_first_step and i == 1)
                or i % int(config.eval_freq) == 0
                or i == int(config.max_timesteps)
            )
            if should_eval:
                print(f"Time steps: {i}")
                eval_info, eval_returns, eval_successes = evaluate(
                    agent=agent,
                    env=eval_env,
                    num_eval_episodes=int(config.n_episodes),
                    eval_temperature=0,
                    seed=int(config.eval_seed),
                )
                eval_log: Dict[str, Any] = {"timestep": int(i)}
                for k, v in eval_info.items():
                    eval_log[f"eval/{k}"] = float(v)
                if eval_returns.size > 0:
                    eval_log["eval/reward_mean"] = float(np.mean(eval_returns))
                    eval_log["eval/reward_std"] = float(np.std(eval_returns))
                else:
                    eval_log["eval/reward_mean"] = float(eval_info.get("episode.return", np.nan))
                    eval_log["eval/reward_std"] = np.nan
                if "episode.normalized_return" in eval_info:
                    eval_log["eval/d4rl_normalized_score_mean"] = float(eval_info["episode.normalized_return"])
                    eval_log["eval/d4rl_normalized_score_std"] = np.nan
                else:
                    eval_log["eval/d4rl_normalized_score_mean"] = np.nan
                    eval_log["eval/d4rl_normalized_score_std"] = np.nan
                if eval_successes.size > 0:
                    eval_log["eval/success_rate"] = float(np.mean(eval_successes))
                    eval_log["eval/success_std"] = float(np.std(eval_successes))
                else:
                    eval_log["eval/success_rate"] = float(eval_info.get("success", np.nan))
                    eval_log["eval/success_std"] = np.nan
    
                upsert_eval_log(eval_logs, eval_log)
                print(
                    f"Evaluation over {config.n_episodes} episodes: "
                    f"reward={eval_log['eval/reward_mean']:.3f} ± {eval_log['eval/reward_std']:.3f}, "
                    f"d4rl_normalized={eval_log['eval/d4rl_normalized_score_mean']:.3f}, "
                    f"success_rate={eval_log['eval/success_rate']:.3f} ± {eval_log['eval/success_std']:.3f}"
                )
    
                if config.log_wandb:
                    wandb_eval_log = {
                        key: to_python_scalar(value)
                        for key, value in eval_log.items()
                        if is_scalar_value(value)
                    }
                    _wandb_log(wandb_eval_log, i)
    
                save_and_upload_eval_logs(eval_logs, config.checkpoints_path, config.log_wandb)
    
                if config.checkpoints_path is not None and config.save_best_model:
                    metric = eval_log["eval/success_rate"]
                    if not np.isfinite(metric):
                        metric = eval_log["eval/d4rl_normalized_score_mean"]
                    if not np.isfinite(metric):
                        metric = eval_log["eval/reward_mean"]
                    if np.isfinite(metric) and metric > best_eval_metric_mean:
                        best_eval_metric_mean = metric
                        save_checkpoint(
                            os.path.join(config.checkpoints_path, "best_checkpoint.pkl"),
                            agent=agent,
                            config=config,
                            log_wandb=config.log_wandb,
                        )
    
            current_timestep = _training_timestep()
            if (
                checkpoint_manager is not None
                and current_timestep % int(config.checkpoint_freq) == 0
            ):
                checkpoint_manager.save_progress(
                    timestep=current_timestep,
                    trainer_state=_progress_state(),
                    eval_logs=eval_logs,
                    status="running",
                )
    except BaseException:
        if checkpoint_manager is not None:
            interrupted_timestep = _training_timestep()
            evaluation_complete = (
                not _evaluation_required(interrupted_timestep)
                or find_eval_log(eval_logs, interrupted_timestep) is not None
            )
            if evaluation_complete:
                checkpoint_manager.save_progress(
                    timestep=interrupted_timestep,
                    trainer_state=_progress_state(),
                    eval_logs=eval_logs,
                    status="interrupted",
                )
                print(f"Saved interrupted checkpoint at timestep {interrupted_timestep}.")
            else:
                print(
                    "Evaluation was interrupted before its result was committed; "
                    "retaining the previous safe checkpoint so evaluation cannot be skipped."
                )
        raise
    final_timestep = _training_timestep()
    if checkpoint_manager is not None:
        final_path = checkpoint_manager.complete(
            timestep=final_timestep,
            final_state=_final_state(),
            save_final_model=bool(config.save_final_model),
            eval_logs=eval_logs,
        )
        if final_path is not None:
            print("---------------------------------------")
            print(f"Saved final checkpoint to: {final_path}")
            print("---------------------------------------")



@pyrallis.wrap()
def train(config: TrainConfig):
    exit_code = 0
    try:
        return _train_impl(config)
    except BaseException:
        exit_code = 1
        raise
    finally:
        if getattr(wandb, "run", None) is not None:
            wandb.finish(exit_code=exit_code)


if __name__ == "__main__":
    train()
