"""FAC (official mathematical implementation) with ORL-SMOOTH experiment plumbing.

The FAC objective, model definitions, optimizer steps, density estimator and
threshold logic are imported from an adapted *namespace-only* copy of the
user-supplied FAC-main.zip. No algorithmic formula is reimplemented here.

Install:
    algorithms/uk_offline/fac_jax_interface.py
    algorithms/uk_offline/fac_official/{agents,utils,envs}/...
    hyperparams/fac_jax.yml

Run from the ORL repository root, which must contain:
    algorithms/uk_offline/common/checkpointing.py

Default behavior: offline training (original 250-epoch flow-proxy pretraining,
original dataset-wide density precomputation, original FAC agent updates).
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

import pickle
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
import jax
import pyrallis
from flax import serialization
from tqdm.auto import trange

# Same repository-root discovery and shared checkpoint API as the user's FQL.
_PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "algorithms").is_dir()
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.uk_offline.common.checkpointing import (  # noqa: E402
    DEFAULT_IDENTITY_IGNORED_FIELDS,
    TrainingCheckpointManager,
    best_eval_metric,
    find_eval_log,
    log_training_exceptions,
    upsert_eval_log,
)
# The following are FAC's actual, unmodified mathematical implementations.
# In the vendored files, ONLY absolute import module paths have been changed.
from algorithms.uk_offline.fac_official.agents.fac import FACAgent  # noqa: E402
from algorithms.uk_offline.fac_official.envs.env_utils import (  # noqa: E402
    make_env_and_datasets,
)
from algorithms.uk_offline.fac_official.utils.datasets import Dataset  # noqa: E402
from algorithms.uk_offline.fac_official.utils.flax_utils import (  # noqa: E402
    restore_only_bcmodel,
    save_only_bcmodel,
)

ALGORITHM_NAME = "FAC"
ALGORITHM_FULL_NAME = "Flow Actor-Critic (official FAC math, ORL-SMOOTH runner)"


@dataclass
class TrainConfig:
    # Experiment and schedule (matches ORL interface).
    device: str = "gpu"
    env: str = "cube-double-play-singletask-v0"
    dataset_name: Optional[str] = None
    seed: int = 0
    eval_seed: Optional[int] = 42
    max_timesteps: int = 1_000_000  # AC updates ONLY; BC proxy is extra.
    eval_freq: int = 100_000
    n_episodes: int = 50

    # Experiment-management I/O.
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    mode: str = "train"    # FAC official offline runner has no actor_refit.
    hyperparams_path: Optional[str] = "hyperparams/fac_jax.yml"
    use_hyperparams: bool = True

    # Data (original FAC values by default).
    batch_size: int = 256
    buffer_size: int = 2_000_000  # Reserved for online extensions, not used offline.
    normalize_states: bool = False
    normalize_reward: bool = False
    action_clip_eps: Optional[float] = 1e-5
    frame_stack: Optional[int] = None
    p_aug: Optional[float] = None

    # Network and optimizer, renamed to the user's conventions.
    lr: Optional[float] = None  # Convenience alias for actor_learning_rate.
    actor_learning_rate: float = 3e-4  # FAC's joint AC optimizer (actor+critic).
    critic_learning_rate: float = 3e-4  # Compatibility: NOT independently optimized.
    bc_learning_rate: float = 3e-4  # FAC's independently optimized flow proxy.
    hidden_dim: int = 512
    actor_n_hiddens: int = 4
    critic_n_hiddens: int = 4
    bc_n_hiddens: int = 4
    actor_hidden_dims: Optional[Tuple[int, ...]] = None
    value_hidden_dims: Optional[Tuple[int, ...]] = None
    bc_hidden_dims: Optional[Tuple[int, ...]] = None
    layer_norm: Optional[bool] = None      # FAC's value_layer_norm.
    actor_layer_norm: Optional[bool] = None
    bc_layer_norm: bool = False
    critic_ln: bool = True
    actor_ln: bool = False
    encoder: Optional[str] = None

    # ORIGINAL FAC algorithm choices. These are mapped directly (no change).
    discount: float = 0.995
    tau: float = 0.005
    num_critics: int = 2
    q_agg: str = "min"  # Bellman target: min or mean; actor Q uses mean in original.
    flow_steps: int = 10
    normalize_q_loss: bool = True
    distil_loss: bool = True
    actor_bc_coef: float = 1.0        # Original: fac_lambda (flow distillation).
    critic_penalty_coef: float = 1.0  # Original: fac_alpha (OOD Q penalty).
    fac_threshold: str = "batch_adaptive"
    logp_method: str = "exact"
    logp_hutch_probes: int = 8
    weight_type: str = "linear"

    # Original FAC proxy stage schedule and pretrained-policy cache.
    epoch_flow_proxy: int = 250
    bc_batch_size: Optional[int] = None  # None: original size/domain-dependent rule.
    pretrained_bc_dir: Optional[str] = "pretrained_bc"

    # ORL logging/checkpoint controls.
    project: str = "ORL-SMOOTH"
    group: str = "FAC-JAX"
    name: str = "FAC-JAX"
    log_wandb: bool = True
    log_every: int = 5_000
    checkpoint_freq: int = 25_000
    save_best_model: bool = True
    save_final_model: bool = True
    eval_at_first_step: bool = True
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        if self.dataset_name is not None:
            self.env = self.dataset_name
        validate_config(self)


def cli_overridden_fields() -> set[str]:
    return {
        arg[2:].split("=", 1)[0].replace("-", "_")
        for arg in sys.argv[1:]
        if arg.startswith("--")
    }


def _tuple(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(int(x) for x in value)
    raise ValueError(f"Hidden dimensions must be a sequence, got {value!r}")


def resolve_config(config: TrainConfig, cli_fields: set[str]) -> TrainConfig:
    """Map ORL names to the original FAC agent settings; resolve after YAML merge."""
    if config.dataset_name is not None:
        config.env = config.dataset_name
    config.actor_hidden_dims = _tuple(config.actor_hidden_dims) or (
        int(config.hidden_dim),
    ) * int(config.actor_n_hiddens)
    config.value_hidden_dims = _tuple(config.value_hidden_dims) or (
        int(config.hidden_dim),
    ) * int(config.critic_n_hiddens)
    config.bc_hidden_dims = _tuple(config.bc_hidden_dims) or (
        int(config.hidden_dim),
    ) * int(config.bc_n_hiddens)
    if config.layer_norm is None:
        config.layer_norm = bool(config.critic_ln)
    if config.actor_layer_norm is None:
        config.actor_layer_norm = bool(config.actor_ln)
    if config.lr is not None:
        # Avoid accidentally overriding an explicit actor_learning_rate in CLI.
        if "actor_learning_rate" not in cli_fields:
            config.actor_learning_rate = float(config.lr)
    else:
        config.lr = float(config.actor_learning_rate)
    refresh_algorithm_names(config)
    validate_config(config)
    if config.critic_learning_rate != config.actor_learning_rate:
        print(
            "Warning: official FAC has ONE joint AC optimizer. "
            f"critic_learning_rate={config.critic_learning_rate} is ignored; "
            f"actor_learning_rate={config.actor_learning_rate} is used for both."
        )
    return config


def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = "ORL-SMOOTH"
    config.group = "FAC-JAX"
    config.name = f"FAC-JAX-{config.env}"
    # Follow IQL/ReBRAC convention: isolate Q aggregation ablations.
    if config.q_agg == "mean":
        config.name += "-QMEAN"
    if config.logp_method != "exact":
        config.name += f"-{config.logp_method.upper()}"
    if config.fac_threshold != "batch_adaptive":
        config.name += f"-{config.fac_threshold.upper()}"


def validate_config(config: TrainConfig) -> None:
    if config.mode != "train":
        raise NotImplementedError("Official FAC ORL wrapper supports mode='train' only.")
    for name in ("max_timesteps", "eval_freq", "n_episodes", "log_every", "checkpoint_freq", "epoch_flow_proxy"):
        if int(getattr(config, name)) < (0 if name in ("max_timesteps", "epoch_flow_proxy") else 1):
            raise ValueError(f"{name} has invalid value {getattr(config, name)!r}")
    if config.batch_size <= 0 or config.hidden_dim <= 0:
        raise ValueError("batch_size and hidden_dim must be positive")
    if config.q_agg not in ("min", "mean"):
        raise ValueError("q_agg must be 'min' or 'mean'")
    if config.num_critics < 1 or config.flow_steps < 1:
        raise ValueError("num_critics/flow_steps must be positive")
    if config.fac_threshold not in ("batch_adaptive", "batch_wide_constant", "dataset_wide_constant"):
        raise ValueError(f"Unsupported FAC threshold: {config.fac_threshold}")
    if config.logp_method not in ("exact", "hutch-rade", "hutch-gaus"):
        raise ValueError(f"Unsupported official FAC logp method: {config.logp_method}")
    if config.weight_type not in ("linear", "logarithmic", "convex", "concave"):
        raise ValueError(f"Unsupported official FAC weight: {config.weight_type}")
    if config.logp_hutch_probes < 1:
        raise ValueError("logp_hutch_probes must be positive")
    if not 0 <= config.discount <= 1 or not 0 <= config.tau <= 1:
        raise ValueError("discount and tau must lie in [0,1]")
    if config.actor_bc_coef < 0 or config.critic_penalty_coef < 0:
        raise ValueError("FAC regularization coefficients must be >= 0")
    if config.normalize_states or config.normalize_reward:
        # No silent preprocessing changes allowed under the exact-FAC contract.
        raise ValueError("Original FAC does not normalize states/rewards; both must be False")
    if config.bc_batch_size is not None and config.bc_batch_size < 1:
        raise ValueError("bc_batch_size must be positive")


def apply_env_hyperparams(config: TrainConfig) -> TrainConfig:
    cli_fields = cli_overridden_fields()
    if not config.use_hyperparams or not config.hyperparams_path:
        return resolve_config(config, cli_fields)
    path = Path(config.hyperparams_path)
    if not path.is_file():
        raise FileNotFoundError(f"FAC hyperparameter YAML was not found: {path}")
    with path.open(encoding="utf-8") as f:
        all_params = yaml.safe_load(f) or {}
    params = all_params.get(config.env)
    if params is None:
        print(f"No FAC YAML entry for {config.env}; using dataclass/CLI settings")
        return resolve_config(config, cli_fields)
    aliases = {
        "fac_alpha": "critic_penalty_coef",
        "fac_lambda": "actor_bc_coef",
        "lr_bc": "bc_learning_rate",
        "value_layer_norm": "layer_norm",
        "n_timesteps": "max_timesteps",
        "offline_steps": "max_timesteps",
        "eval_interval": "eval_freq",
        "eval_episodes": "n_episodes",
        "log_interval": "log_every",
        "env_name": "env",
        "learning_rate": "lr",
    }
    fields = set(TrainConfig.__dataclass_fields__)
    applied, skipped = [], []
    for raw, value in params.items():
        field = aliases.get(raw, raw)
        if field not in fields:
            print(f"Warning: unrecognized FAC YAML parameter {raw!r}, ignoring")
            continue
        if field in cli_fields or raw in cli_fields:
            skipped.append(field)
            continue
        if field in ("actor_hidden_dims", "value_hidden_dims", "bc_hidden_dims"):
            value = _tuple(value)
        if isinstance(value, float) and value.is_integer() and field in (
            "max_timesteps", "eval_freq", "n_episodes", "log_every", "checkpoint_freq", "epoch_flow_proxy", "batch_size"
        ):
            value = int(value)
        setattr(config, field, value)
        applied.append(field)
    if applied:
        print(f"Applied FAC YAML ({config.env}): {', '.join(applied)}")
    if skipped:
        print(f"Keeping explicit FAC CLI overrides: {', '.join(skipped)}")
    return resolve_config(config, cli_fields)


def agent_config(config: TrainConfig) -> dict:
    """EXACT mapping to FAC-main/agents/fac.py:get_config, no loss modifications."""
    return dict(
        agent_name="fac",
        lr=float(config.actor_learning_rate),
        lr_bc=float(config.bc_learning_rate),
        batch_size=int(config.batch_size),
        actor_hidden_dims=tuple(config.actor_hidden_dims),
        value_hidden_dims=tuple(config.value_hidden_dims),
        bc_hidden_dims=tuple(config.bc_hidden_dims),
        value_layer_norm=bool(config.layer_norm),
        actor_layer_norm=bool(config.actor_layer_norm),
        bc_layer_norm=bool(config.bc_layer_norm),
        flow_steps=int(config.flow_steps),
        normalize_q_loss=bool(config.normalize_q_loss),
        distil_loss=bool(config.distil_loss),
        fac_alpha=float(config.critic_penalty_coef),
        fac_lambda=float(config.actor_bc_coef),
        fac_threshold=str(config.fac_threshold),
        logp_method=str(config.logp_method),
        logp_hutch_probes=int(config.logp_hutch_probes),
        num_critics=int(config.num_critics),
        discount=float(config.discount),
        tau=float(config.tau),
        q_agg=str(config.q_agg),
        weight_type=str(config.weight_type),
        encoder=config.encoder,
    )


def finalize_checkpoint_path(config: TrainConfig) -> TrainConfig:
    if config.checkpoints_path is not None:
        config.checkpoints_path = str(Path(config.checkpoints_path) / config.name / str(config.seed))
    return config


def select_jax_device(name: str):
    try:
        dev = jax.devices("gpu" if name.lower() == "cuda" else name.lower())[0]
    except Exception:
        print(f"Requested JAX device {name!r} unavailable; falling back")
        dev = jax.devices()[0]
    print(f"Using JAX device: {dev}")
    return dev


def set_seed(seed: int, env=None) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if env is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
        for space in (env.action_space, env.observation_space):
            if hasattr(space, "seed"):
                space.seed(seed)


def _save_pickle(path: Union[str, Path], obj: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def _load_pickle(path: Union[str, Path]):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_logs_npz(logs: List[dict], save_dir: Optional[str]):
    if not save_dir or not logs:
        return
    keys = sorted({key for row in logs for key in row})
    data = {}
    for key in keys:
        values = [row.get(key, np.nan) for row in logs]
        try:
            data[key] = np.asarray(values, dtype=np.float64 if key != "timestep" else np.int64)
        except (TypeError, ValueError):
            data[key] = np.asarray(values, dtype=object)
    np.savez(Path(save_dir) / "eval_logs.npz", **data)


def resolve_checkpoint_path(load_model, run_name=None, seed=None) -> Path:
    root = Path(load_model)
    if root.is_file():
        return root
    if not root.exists():
        raise FileNotFoundError(f"Model path does not exist: {root}")
    for candidate_root in (root, root / str(run_name) / str(seed)):
        for filename in ("checkpoint.pkl", "best_checkpoint.pkl"):
            candidate = candidate_root / filename
            if candidate.is_file():
                return candidate
    files = sorted(set(root.glob("*/checkpoint.pkl")) | set(root.glob("*/*/checkpoint.pkl")))
    if len(files) == 1:
        return files[0]
    if not files:
        raise FileNotFoundError(f"No checkpoint.pkl found under {root}")
    raise ValueError(f"Multiple checkpoints under {root}; specify exact file:\n" + "\n".join(map(str, files)))


def _agent_payload(agent, config: TrainConfig, ac_step: int, phase: str) -> dict:
    return {
        "agent": serialization.to_state_dict(agent),
        "config": asdict(config),
        "_training_timestep": int(ac_step),
        "_fac_phase": str(phase),
        "_numpy_rng_state": np.random.get_state(),
        "_python_rng_state": random.getstate(),
    }


def _restore_agent(agent, payload):
    state = payload["agent"] if isinstance(payload, dict) and "agent" in payload else payload
    return serialization.from_state_dict(agent, state)


def _proxy_batch_size(config: TrainConfig, dataset_size: int) -> int:
    if config.bc_batch_size is not None:
        return int(config.bc_batch_size)
    if "visual" in config.env:
        return int(config.batch_size)
    # Exact FAC-main/main.py dataset-size rule.
    factor = 1 if dataset_size < 100_000 else 4 if dataset_size < 500_000 else 16
    return int(config.batch_size) * factor


def _eval_metrics(agent, env, episodes: int, eval_seed: Optional[int]) -> dict:
    """Original action sampling/reset semantics; adds per-episode statistics."""
    from algorithms.uk_offline.fac_official.utils.evaluation import flatten
    from collections import defaultdict

    # The official FAC evaluator draws a fresh seed from NumPy per evaluation.
    # An explicit eval_seed reproduces the user's FQL-JAX deterministic RNG API.
    rng_seed = (
        int(np.random.randint(0, 2 ** 32)) if eval_seed is None else int(eval_seed)
    )
    key = jax.random.PRNGKey(rng_seed)
    info_values = defaultdict(list)
    rewards, successes, normalized = [], [], []
    for _ in trange(episodes, desc="FAC Eval", leave=False):
        obs, info = env.reset()
        total_return, done = 0.0, False
        last_success = np.nan
        while not done:
            key, action_key = jax.random.split(key)
            action = np.asarray(agent.sample_actions(observations=obs, seed=action_key, temperature=0))
            action = np.clip(action, -1, 1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_return += float(reward)
            if "success" in info:
                last_success = float(np.asarray(info["success"]).reshape(-1)[0])
        flat = flatten(info)
        for k, v in flat.items():
            if isinstance(v, (int, float, bool, np.number)):
                info_values[k].append(float(v))
        rewards.append(float(flat.get("episode.return", total_return)))
        successes.append(float(flat.get("success", flat.get("episode.success", last_success))))
        normalized.append(float(flat.get("episode.normalized_return", np.nan)))

    def avg_std(values):
        x = np.asarray(values, dtype=np.float64)
        x = x[np.isfinite(x)]
        return (float(x.mean()), float(x.std())) if x.size else (float("nan"), float("nan"))

    reward_mean, reward_std = avg_std(rewards)
    success_mean, success_std = avg_std(successes)
    norm_mean, norm_std = avg_std(normalized)
    metrics = {
        "eval/reward_mean": reward_mean,
        "eval/reward_std": reward_std,
        "eval/success_rate": success_mean,
        "eval/success_std": success_std,
        "eval/d4rl_normalized_score_mean": norm_mean,
        "eval/d4rl_normalized_score_std": norm_std,
    }
    metrics.update({f"eval/{k}": float(np.mean(v)) for k, v in info_values.items() if v})
    return metrics


def _save_density_cache(path: Path, train_ds, val_ds, config: TrainConfig):
    """Persist Hutchinson-generated densities: AC resumes must not resample them."""
    metadata = {
        "env": config.env,
        "seed": int(config.seed),
        "logp_method": config.logp_method,
        "flow_steps": int(config.flow_steps),
        "logp_hutch_probes": int(config.logp_hutch_probes),
        "train_count": int(train_ds.size),
        "val_count": int(val_ds.size) if val_ds is not None else 0,
    }
    payload = {"metadata": np.asarray(yaml.safe_dump(metadata)),
               "train_logp": train_ds["estimated_logp"][:train_ds.size],
               "train_logp_min": np.asarray(train_ds.estimated_logp_min, dtype=np.float32)}
    if val_ds is not None:
        payload["val_logp"] = val_ds["estimated_logp"][:val_ds.size]
        payload["val_logp_min"] = np.asarray(val_ds.estimated_logp_min, dtype=np.float32)
    temp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temp, **payload)
    os.replace(temp, path)


def _load_density_cache(path: Path, train_ds, val_ds, config: TrainConfig) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as z:
        metadata = yaml.safe_load(str(z["metadata"].item()))
        expected = dict(
            env=config.env,
            seed=int(config.seed),
            logp_method=config.logp_method,
            flow_steps=int(config.flow_steps),
            logp_hutch_probes=int(config.logp_hutch_probes),
            train_count=int(train_ds.size),
            val_count=int(val_ds.size) if val_ds is not None else 0,
        )
        if metadata != expected:
            print("FAC density-cache metadata differs; recomputing densities")
            return False
        def attach(ds, values, minimum):
            # Original FAC dataset uses a mutable internal dictionary.
            n = getattr(ds, "max_size", ds.size)
            data = np.full((n,), np.nan, dtype=np.float32)
            data[:ds.size] = values.astype(np.float32)
            ds._dict["estimated_logp"] = data
            ds._dict["estimated_logp_min"] = np.full((n,), minimum, dtype=np.float32)
            ds.estimated_logp_min = float(minimum)
        attach(train_ds, z["train_logp"], float(z["train_logp_min"].item()))
        if val_ds is not None:
            attach(val_ds, z["val_logp"], float(z["val_logp_min"].item()))
        return True


def _prepare_density_cache(agent, train_ds, val_ds, config: TrainConfig, run_dir: Optional[Path], bc_batch: int):
    cache = run_dir / "fac_density_cache.npz" if run_dir is not None else None
    if cache is not None and _load_density_cache(cache, train_ds, val_ds, config):
        print(f"Loaded FAC behavior log-densities: {cache}")
        return
    print(f"Computing original FAC {config.logp_method} flow densities")
    train_ds.compute_and_attach_estimated_logp(
        agent=agent, method=config.logp_method, batch_size=bc_batch
    )
    if val_ds is not None:
        val_ds.compute_and_attach_estimated_logp(
            agent=agent, method=config.logp_method, batch_size=bc_batch
        )
    if cache is not None:
        _save_density_cache(cache, train_ds, val_ds, config)
        print(f"Saved FAC behavior log-densities: {cache}")


def _train_impl(config: TrainConfig):
    config = finalize_checkpoint_path(apply_env_hyperparams(config))
    run_dir = Path(config.checkpoints_path) if config.checkpoints_path else None

    checkpoint_manager, preparation = None, None
    if run_dir is not None:
        checkpoint_manager = TrainingCheckpointManager(
            run_dir=str(run_dir),
            current_config=asdict(config),
            default_config=asdict(TrainConfig()),
            max_timesteps=int(config.max_timesteps),
            checkpoint_type="fac_jax_training_progress",
            identity_ignored_fields=DEFAULT_IDENTITY_IGNORED_FIELDS,
            checkpoint_version=2,
            accepted_checkpoint_versions=(1, 2),
            wandb_enabled=bool(config.log_wandb),
            wandb_entity=config.wandb_entity,
            wandb_project=config.project,
            final_checkpoint_name="checkpoint.pkl",
        )
        preparation = checkpoint_manager.prepare()
        print(preparation.message)
        if preparation.is_completed:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        saved_config = run_dir / "config.yaml"
        if not saved_config.exists():
            with saved_config.open("w", encoding="utf-8") as f:
                yaml.safe_dump(asdict(config), f, allow_unicode=True, sort_keys=True)

    dev = select_jax_device(config.device)
    # Original FAC loaders/wrappers: prevents any silent reward/observation change.
    _, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        config.env,
        frame_stack=config.frame_stack,
        action_clip_eps=config.action_clip_eps,
    )
    train_dataset = Dataset.create(**dict(train_dataset))
    # FAC originally copies the dataset into a large replay buffer even in offline
    # training. This version samples the identical dataset without an unnecessary
    # 2M-capacity copy (critical for pixel tasks); sampling and losses are unchanged.
    for ds in (train_dataset, val_dataset):
        if ds is not None:
            ds.p_aug = config.p_aug
            ds.frame_stack = config.frame_stack
            ds.return_next_actions = False
    set_seed(config.seed, eval_env)

    print(f"Training {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {config.seed}")
    example = train_dataset.sample(1)
    agent = FACAgent.create(
        config.seed, example["observations"], example["actions"], agent_config(config)
    )
    agent = jax.device_put(agent, dev)
    ac_step = 0
    eval_logs: List[dict] = []

    def progress_state():
        return _agent_payload(agent, config, ac_step, phase="ac")

    def load_progress(payload):
        nonlocal agent, ac_step
        agent = jax.device_put(_restore_agent(agent, payload), dev)
        ac_step = int(payload.get("_training_timestep", 0))
        if "_numpy_rng_state" in payload:
            np.random.set_state(payload["_numpy_rng_state"])
        if "_python_rng_state" in payload:
            random.setstate(payload["_python_rng_state"])

    def current_step():
        return int(ac_step)

    # An explicit --load_model is a full agent restore, NOT a from-scratch refit.
    previous_file = None
    if config.load_model:
        previous_file = resolve_checkpoint_path(config.load_model, config.name, config.seed)
        print(f"Loading FAC agent from {previous_file}")
        load_progress(_load_pickle(previous_file))
        # Preserve original Hutchinson estimates across manual checkpoint loads.
        if run_dir is not None:
            source_cache = previous_file.parent / "fac_density_cache.npz"
            target_cache = run_dir / "fac_density_cache.npz"
            if source_cache.is_file() and not target_cache.is_file():
                import shutil
                shutil.copy2(source_cache, target_cache)

    if checkpoint_manager is not None and preparation.is_resuming:
        ac_step, eval_logs, _ = checkpoint_manager.restore(
            load_trainer_state=load_progress,
            get_restored_timestep=current_step,
        )
        print(f"Restored FAC AC training at timestep {ac_step}")
    elif previous_file is not None:
        # Loading a full checkpoint must NOT retrain its already learned BC proxy.
        bc_batch = _proxy_batch_size(config, train_dataset.size)
        _prepare_density_cache(agent, train_dataset, val_dataset, config, run_dir, bc_batch)
        if checkpoint_manager is not None:
            checkpoint_manager.save_progress(
                timestep=ac_step, trainer_state=progress_state(),
                eval_logs=eval_logs, status="running"
            )
    else:
        # Only execute the official multi-stage initialization for fresh runs.
        bc_batch = _proxy_batch_size(config, train_dataset.size)
        bc_iters_per_epoch = (train_dataset.size + bc_batch - 1) // bc_batch
        bc_total_iters = int(config.epoch_flow_proxy) * bc_iters_per_epoch
        bc_step = 0
        stage_path = run_dir / "fac_bc_progress.pkl" if run_dir else None
        cache_root = Path(config.pretrained_bc_dir) if config.pretrained_bc_dir else None
        cache_folder = cache_root / f"{config.env}_seed_{config.seed}" if cache_root else None

        should_train_bc = True
        if stage_path and stage_path.is_file():
            stage = _load_pickle(stage_path)
            if (stage.get("bc_total_iters"), stage.get("bc_batch_size")) != (bc_total_iters, bc_batch):
                raise ValueError("FAC BC resume config mismatch: batch size or proxy-epoch count changed")
            agent = jax.device_put(_restore_agent(agent, stage), dev)
            bc_step = int(stage["bc_iteration"])
            np.random.set_state(stage["_numpy_rng_state"])
            random.setstate(stage["_python_rng_state"])
            print(f"Restoring original FAC BC flow from iteration {bc_step}/{bc_total_iters}")
        elif cache_folder and cache_folder.is_dir():
            print(f"Reusing official pretrained FAC BC flow: {cache_folder}")
            agent = restore_only_bcmodel(agent, str(cache_root), config.env, str(config.seed))
            should_train_bc = False

        if should_train_bc:
            print(f"Training original FAC BC proxy: {bc_total_iters} updates, batch_size={bc_batch}")
            def save_bc_stage():
                if stage_path is None:
                    return
                state = _agent_payload(agent, config, ac_step=0, phase="bc")
                state.update(bc_iteration=bc_step,
                             bc_batch_size=bc_batch, bc_total_iters=bc_total_iters)
                _save_pickle(stage_path, state)
            try:
                for iteration in trange(bc_step + 1, bc_total_iters + 1, desc="FAC Train BC"):
                    batch = train_dataset.sample(bc_batch)
                    agent, bc_log = agent.update(batch, mode="train_bc")
                    bc_step = iteration
                    if stage_path and iteration % config.checkpoint_freq == 0:
                        save_bc_stage()
            except BaseException:
                save_bc_stage()
                raise
            if cache_root is not None:
                save_only_bcmodel(agent, str(cache_root), config.env, str(config.seed))

        _prepare_density_cache(agent, train_dataset, val_dataset, config, run_dir, bc_batch)
        if stage_path and stage_path.exists():
            stage_path.unlink()
        if checkpoint_manager is not None:
            checkpoint_manager.save_progress(
                timestep=ac_step, trainer_state=progress_state(),
                eval_logs=eval_logs, status="running"
            )

    # If recovering an AC progress checkpoint, log-densities need to be restored
    # separately: they are computed ONCE in official FAC, not continually refreshed.
    if checkpoint_manager is not None and preparation.is_resuming:
        bc_batch = _proxy_batch_size(config, train_dataset.size)
        _prepare_density_cache(agent, train_dataset, val_dataset, config, run_dir, bc_batch)

    # ORL W&B lifecycle: resume-aware when using shared checkpoint manager.
    if config.log_wandb:
        try:
            import wandb
        except ImportError:
            print("Warning: W&B unavailable; continuing local FAC training")
            config.log_wandb = False

    if config.log_wandb:
        if checkpoint_manager is not None:
            if not checkpoint_manager.initialize_wandb_with_fallback(
                wandb_module=wandb,
                config=asdict(config),
                code_root=_PROJECT_ROOT,
            ):
                config.log_wandb = False
        else:
            try:
                import wandb
                run = wandb.init(config=asdict(config), project=config.project,
                                 group=config.group, name=config.name, id=str(uuid.uuid4()))
                if run is not None and hasattr(run, "log_code"):
                    run.log_code(str(_PROJECT_ROOT))
            except Exception as exc:
                print(f"Warning: W&B unavailable: {exc}; local-only training")
                config.log_wandb = False

    def wandb_log(metrics, step):
        if not config.log_wandb:
            return
        if checkpoint_manager is not None:
            checkpoint_manager.log_wandb(metrics, int(step))
        else:
            import wandb
            wandb.log(metrics, step=int(step))

    best_value = best_eval_metric(eval_logs)
    last_update_info: dict = {}
    start_time = last_log_time = time.time()

    def eval_is_due(step):
        return ((config.eval_at_first_step and step == 1)
                or step % config.eval_freq == 0
                or step == config.max_timesteps)

    try:
        for step in trange(ac_step + 1, int(config.max_timesteps) + 1, desc="FAC Train AC"):
            batch = train_dataset.sample(config.batch_size)
            agent, info = agent.update(batch, mode="train_ac", use_cp=True, online_finetuning=False)
            ac_step = step
            last_update_info = {k: float(jax.device_get(v)) for k, v in info.items()}

            if step % config.log_every == 0:
                log = {f"training/{k}": value for k, value in last_update_info.items()}
                if val_dataset is not None:
                    val = val_dataset.sample(config.batch_size)
                    _, validation_info = agent.total_loss(val, grad_params=None, mode="train_ac")
                    log.update({f"validation/{k}": float(jax.device_get(v)) for k, v in validation_info.items()})
                log["time/epoch_time"] = (time.time() - last_log_time) / config.log_every
                log["time/total_time"] = time.time() - start_time
                last_log_time = time.time()
                wandb_log(log, step)

            if eval_is_due(step):
                eval_log = {"timestep": int(step)}
                eval_log.update(_eval_metrics(agent, eval_env, config.n_episodes, config.eval_seed))
                upsert_eval_log(eval_logs, eval_log)
                print(
                    f"[{step}] reward={eval_log['eval/reward_mean']:.3f} "
                    f"± {eval_log['eval/reward_std']:.3f}; "
                    f"success={eval_log['eval/success_rate']:.3f} "
                    f"± {eval_log['eval/success_std']:.3f}"
                )
                _save_logs_npz(eval_logs, config.checkpoints_path)
                wandb_log({k: v for k, v in eval_log.items() if np.isscalar(v)}, step)
                metric = eval_log["eval/success_rate"]
                if not np.isfinite(metric):
                    metric = eval_log["eval/d4rl_normalized_score_mean"]
                if not np.isfinite(metric):
                    metric = eval_log["eval/reward_mean"]
                if (run_dir is not None and config.save_best_model
                        and np.isfinite(metric) and metric > best_value):
                    best_value = metric
                    _save_pickle(run_dir / "best_checkpoint.pkl", progress_state())

            if checkpoint_manager is not None and step % config.checkpoint_freq == 0:
                checkpoint_manager.save_progress(
                    timestep=step, trainer_state=progress_state(),
                    eval_logs=eval_logs, status="running"
                )
    except BaseException:
        if checkpoint_manager is not None:
            if not eval_is_due(ac_step) or find_eval_log(eval_logs, ac_step) is not None:
                checkpoint_manager.save_progress(
                    timestep=ac_step, trainer_state=progress_state(),
                    eval_logs=eval_logs, status="interrupted"
                )
                print(f"Saved FAC interrupted checkpoint at {ac_step}")
            else:
                print("FAC evaluation incomplete; retaining last safe progress checkpoint")
        raise

    if checkpoint_manager is not None:
        final_path = checkpoint_manager.complete(
            timestep=ac_step, final_state=progress_state(),
            save_final_model=bool(config.save_final_model), eval_logs=eval_logs
        )
        print(f"FAC complete: {final_path}")


@pyrallis.wrap()
@log_training_exceptions
def train(config: TrainConfig):
    exit_code = 0
    try:
        return _train_impl(config)
    except BaseException:
        exit_code = 1
        raise
    finally:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish(exit_code=exit_code)
        except ImportError:
            pass


if __name__ == "__main__":
    train()
