"""Official Guided Flow Policy (GFP) with ORL-SMOOTH experiment management.

The mathematical GFP implementation is imported unmodified (except package import
paths) from the user-supplied guided-flow-policy-main.zip:
  algorithms/uk_offline/gfp_official/agents/gfp.py
  algorithms/uk_offline/gfp_official/utils/{networks,encoders,flax_utils,datasets}.py

This runner only changes the experiment surface:
- Pyrallis + TrainConfig, environment YAML, explicit CLI override
- ORL-SMOOTH run name, path, W&B, per-episode evaluation logs
- common TrainingCheckpointManager with automatic resume
- FQL-style best/final model checkpoints and manual --load_model

Default training uses the original GFP CPU/visual or fused-GPU/state update
path; no losses, critic aggregation, flow matching, guidance, architecture,
optimizer or optimizer-update sequences are reimplemented here.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import copy
import pickle
import random
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import numpy as np
import pyrallis
import yaml
from flax import serialization
from tqdm.auto import trange

try:
    import wandb
except ImportError:
    class _UnavailableWandb:
        run = None

        def init(self, *args, **kwargs):
            raise ImportError("wandb unavailable: install wandb or run with --log_wandb False")

        def log(self, *args, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

    wandb = _UnavailableWandb()

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
from algorithms.uk_offline.gfp_official.agents.gfp import GFPAgent  # noqa: E402
from algorithms.uk_offline.gfp_official.utils.datasets import (  # noqa: E402
    Dataset, ReplayBuffer,
)
from algorithms.uk_offline.gfp_official.envs.env_utils import (  # noqa: E402
    make_env_and_datasets as original_make_env_and_datasets,
)

ALGORITHM_NAME = "GFP"
ALGORITHM_FULL_NAME = "Guided Flow Policy — official algorithm, ORL-SMOOTH runner"


@dataclass
class TrainConfig:
    # Experiment.
    device: str = "gpu"
    env: str = "cube-double-play-singletask-v0"
    dataset_name: Optional[str] = None
    seed: int = 0
    eval_seed: int = 42

    # Schedule.
    max_timesteps: int = 1_000_000
    eval_freq: int = 100_000
    n_episodes: int = 50
    eval_at_first_step: bool = True
    eval_flow_policy: bool = True

    # I/O and modes.
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    mode: str = "train"  # GFP original has no actor-only refit stage.
    hyperparams_path: Optional[str] = "hyperparams/gfp_jax.yml"
    use_hyperparams: bool = True
    run_tag: str = ""  # optional to isolate coefficient sweeps of same env/seed

    # Dataset / sampling.
    batch_size: int = 256
    buffer_size: int = 2_000_000
    normalize_states: bool = False  # official GFP does not perform state normalization
    normalize_reward: bool = False  # official GFP does not perform reward normalization
    action_clip_eps: Optional[float] = 1e-5
    frame_stack: Optional[int] = None
    p_aug: Optional[float] = None
    dataset_on_gpu: bool = True   # official state-based GFP default
    n_steps_fused: int = 4        # official state-based GFP default

    # Original GFP's shared Adam optimizer.
    lr: Optional[float] = None
    actor_learning_rate: float = 3e-4  # alias for shared lr, not separate actor tx
    critic_learning_rate: float = 3e-4  # accepted for ORL naming; shared optimizer
    hidden_dim: int = 512
    actor_n_hiddens: int = 4
    critic_n_hiddens: int = 4
    actor_hidden_dims: Optional[Tuple[int, ...]] = None
    value_hidden_dims: Optional[Tuple[int, ...]] = None
    layer_norm: Optional[bool] = None
    actor_layer_norm: Optional[bool] = None
    critic_ln: bool = True
    actor_ln: bool = False
    num_critics: int = 2  # official GFP hardcodes 2
    encoder: Optional[str] = None

    # ORIGINAL GFP algorithm knobs, unchanged and passed through by name.
    discount: float = 0.99
    tau: float = 0.005
    q_agg: str = "mean"         # critic ensemble aggregation in original GFP
    target_agg: str = "actor"   # reference in Bellman target
    guidance_agg: str = "actor" # reference in VaBC guidance
    guidance_fn: str = "softmax"
    target_use_next: bool = False  # official GFP default (intentionally False)
    alpha: float = 1.0            # flow-to-actor distillation coefficient
    eta_temperature: float = 1e-3 # value-aware BC guidance temperature
    eta_temperature_online: Optional[float] = None
    flow_steps: int = 10
    encode_time_dim: int = 64
    normalize_q_loss: bool = True
    log_metrics: bool = True

    # ORL-SMOOTH logging / checkpointing.
    project: str = "ORL-SMOOTH"
    group: str = "GFP-JAX"
    name: str = "GFP-JAX"
    log_wandb: bool = True
    log_every: int = 5_000
    checkpoint_freq: int = 25_000
    save_best_model: bool = True
    save_final_model: bool = True
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        if self.dataset_name is not None:
            self.env = self.dataset_name
        if self.mode != "train":
            raise NotImplementedError("Original GFP ORL wrapper implements mode='train' only")


def _cli_overridden_fields() -> set[str]:
    return {
        a[2:].split("=", 1)[0].replace("-", "_")
        for a in sys.argv[1:] if a.startswith("--")
    }


def _coerce_hparam_value(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return tuple(value)
    return value


def _tuple_or_default(dims: Any, dim: int, n: int) -> Tuple[int, ...]:
    if dims is None:
        return tuple([int(dim)] * int(n))
    if not isinstance(dims, (tuple, list)):
        raise ValueError(f"Expected list/tuple for dims, got {dims!r}")
    return tuple(int(x) for x in dims)


def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = "ORL-SMOOTH"
    config.group = "GFP-JAX"
    # Sanitize Minari's slash-separated IDs; do NOT change OGBench names.
    safe_env = re.sub(r"[^a-zA-Z0-9_.-]+", "_", config.env)
    parts = [f"GFP-JAX-{safe_env}"]
    # Preserve the ORL distinction between different aggregation experiments.
    if config.q_agg != "mean":
        parts.append("Q" + config.q_agg.upper())
    if config.target_agg != "actor":
        parts.append("T" + config.target_agg.upper())
    if config.guidance_agg != "actor":
        parts.append("G" + config.guidance_agg.upper())
    if config.flow_steps != 10:
        parts.append("F" + str(config.flow_steps))
    if config.run_tag:
        parts.append(re.sub(r"[^a-zA-Z0-9_.-]+", "_", config.run_tag))
    config.name = "-".join(parts)


def validate_config(config: TrainConfig) -> None:
    if config.mode != "train":
        raise NotImplementedError("Original GFP does not implement actor_refit")
    for k in ("eval_freq", "n_episodes", "log_every", "checkpoint_freq", "batch_size", "flow_steps", "n_steps_fused", "hidden_dim", "actor_n_hiddens", "critic_n_hiddens", "encode_time_dim"):
        if int(getattr(config, k)) <= 0:
            raise ValueError(f"{k} must be > 0")
    if config.max_timesteps < 0 or config.buffer_size <= 0:
        raise ValueError("max_timesteps must be nonnegative, buffer_size must be positive")
    if config.num_critics != 2:
        raise ValueError("Official GFP always uses 2 critics; num_critics must remain 2")
    if config.q_agg not in ("min", "mean"):
        raise ValueError("q_agg must be 'min' or 'mean'")
    for k in ("target_agg", "guidance_agg"):
        if getattr(config, k) not in ("actor", "vabc", "min", "mean", "max"):
            raise ValueError(f"Unsupported {k}: {getattr(config, k)!r}")
    if config.guidance_fn not in ("softmax", "advantage"):
        raise ValueError("guidance_fn must be 'softmax' or 'advantage'")
    if config.eta_temperature < 0 or config.alpha < 0:
        raise ValueError("alpha and eta_temperature must be >= 0")
    if not (0 <= config.discount <= 1 and 0 <= config.tau <= 1):
        raise ValueError("discount and tau must be in [0,1]")
    if config.lr is None or config.lr <= 0:
        raise ValueError("lr must be > 0")
    if config.normalize_states or config.normalize_reward:
        raise ValueError("To preserve the ORIGINAL GFP data path, normalize_states/reward must be False")
    # Original GPU fused loop only runs at multiples of n_steps_fused.
    if config.dataset_on_gpu:
        for k in ("max_timesteps", "eval_freq", "checkpoint_freq"):
            if getattr(config, k) % config.n_steps_fused:
                raise ValueError(f"With dataset_on_gpu=True, {k} must be divisible by n_steps_fused")


def resolve_config(config: TrainConfig, cli_fields: set[str]) -> TrainConfig:
    if config.dataset_name is not None:
        config.env = config.dataset_name
    # ORL-friendly 'flow' spelling maps to the original code's 'vabc'.
    if config.target_agg == "flow":
        config.target_agg = "vabc"
    if config.guidance_agg == "flow":
        config.guidance_agg = "vabc"
    config.actor_hidden_dims = _tuple_or_default(
        config.actor_hidden_dims, config.hidden_dim, config.actor_n_hiddens
    )
    config.value_hidden_dims = _tuple_or_default(
        config.value_hidden_dims, config.hidden_dim, config.critic_n_hiddens
    )
    if config.layer_norm is None:
        config.layer_norm = bool(config.critic_ln)
    if config.actor_layer_norm is None:
        config.actor_layer_norm = bool(config.actor_ln)

    # One shared GFP optimizer: explicit CLI --lr and --actor_learning_rate
    # both override the YAML shared lr (with --lr winning if both are set).
    if "lr" in cli_fields and config.lr is not None:
        pass
    elif "actor_learning_rate" in cli_fields:
        config.lr = float(config.actor_learning_rate)
    elif config.lr is None:
        config.lr = float(config.actor_learning_rate)
    config.actor_learning_rate = float(config.lr)
    if float(config.critic_learning_rate) != float(config.lr):
        print("Warning: official GFP has one shared Adam optimizer; "
              f"critic_learning_rate={config.critic_learning_rate} is ignored, using lr={config.lr}.")

    # The original main.py forces visual observations onto CPU dataset sampling.
    if config.env.startswith("visual-"):
        if config.dataset_on_gpu:
            print("Original GFP visual configuration: disabling dataset_on_gpu")
            config.dataset_on_gpu = False
        if config.frame_stack is None:
            config.frame_stack = 3
        if config.p_aug is None:
            config.p_aug = 0.5
        if config.encoder is None:
            config.encoder = "impala_small"

    refresh_algorithm_names(config)
    validate_config(config)
    return config


def apply_env_hyperparams(config: TrainConfig) -> TrainConfig:
    cli_fields = _cli_overridden_fields()
    if config.dataset_name is not None:
        config.env = config.dataset_name
    if not config.use_hyperparams or config.hyperparams_path is None:
        return resolve_config(config, cli_fields)
    path = Path(config.hyperparams_path)
    if not path.is_file():
        raise FileNotFoundError(f"Hyperparameter YAML missing: {path}")
    with path.open(encoding="utf-8") as f:
        all_params = yaml.safe_load(f) or {}
    selected = all_params.get(config.env)
    if selected is None:
        print(f"No GFP YAML entry for '{config.env}' — using dataclass/CLI values")
        return resolve_config(config, cli_fields)
    aliases = {
        "offline_steps": "max_timesteps", "n_timesteps": "max_timesteps",
        "eval_interval": "eval_freq", "eval_episodes": "n_episodes",
        "env_name": "env", "learning_rate": "lr",
        "n_fused_steps": "n_steps_fused",
    }
    fields = set(TrainConfig.__dataclass_fields__)
    applied, skipped_cli, skipped_unknown = [], [], []
    for old_key, raw_value in selected.items():
        key = aliases.get(old_key, old_key)
        if key not in fields:
            skipped_unknown.append(old_key)
            continue
        if key in cli_fields or old_key in cli_fields:
            skipped_cli.append(old_key)
            continue
        # An explicit --actor_learning_rate also overrides YAML 'lr'.
        if key == "lr" and "actor_learning_rate" in cli_fields:
            skipped_cli.append(old_key)
            continue
        setattr(config, key, _coerce_hparam_value(raw_value))
        applied.append(key)
    config = resolve_config(config, cli_fields)
    if applied:
        print(f"Loaded GFP hyperparameters for {config.env} from {path}: {', '.join(applied)}")
    if skipped_cli:
        print("Kept explicit CLI overrides: " + ", ".join(skipped_cli))
    if skipped_unknown:
        print("Ignored unknown keys in GFP YAML: " + ", ".join(skipped_unknown))
    return config


def finalize_checkpoint_path(config: TrainConfig) -> TrainConfig:
    if config.checkpoints_path is not None:
        config.checkpoints_path = os.path.join(config.checkpoints_path, config.name, str(config.seed))
    return config


def select_jax_device(device: str):
    backend = "gpu" if device.lower() == "cuda" else device.lower()
    try:
        out = jax.devices(backend)[0]
    except Exception:
        print(f"Requested device '{device}' unavailable; falling back to default JAX device")
        out = jax.devices()[0]
    print(f"Using JAX device: {out}")
    return out


def set_seed(seed: int, env: Any = None):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if env is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
        for space_name in ("action_space", "observation_space"):
            space = getattr(env, space_name, None)
            if space is not None and hasattr(space, "seed"):
                space.seed(seed)


def agent_config(config: TrainConfig) -> Dict[str, Any]:
    """Only renamed ORL fields are mapped; algorithm arguments mirror original config/agent/gfp.yaml."""
    return {
        "agent_name": "gfp",
        "lr": float(config.lr),
        "batch_size": int(config.batch_size),
        "actor_hidden_dims": tuple(config.actor_hidden_dims),
        "value_hidden_dims": tuple(config.value_hidden_dims),
        "layer_norm": bool(config.layer_norm),
        "actor_layer_norm": bool(config.actor_layer_norm),
        "discount": float(config.discount),
        "tau": float(config.tau),
        "encoder": config.encoder,
        "alpha": float(config.alpha),
        "eta_temperature": float(config.eta_temperature),
        "eta_temperature_online": config.eta_temperature_online,
        "q_agg": config.q_agg,
        "target_agg": config.target_agg,
        "guidance_agg": config.guidance_agg,
        "guidance_fn": config.guidance_fn,
        "target_use_next": bool(config.target_use_next),
        "flow_steps": int(config.flow_steps),
        "normalize_q_loss": bool(config.normalize_q_loss),
        "encode_time_dim": int(config.encode_time_dim),
        "log_metrics": bool(config.log_metrics),
    }


def _benchmark(env: str) -> str:
    if "singletask" in env:
        return "OGBench"
    if env.startswith(("D4RL/", "mujoco/")):
        return "Minari"
    return "D4RL"


def make_env_and_datasets(config: TrainConfig):
    """Use official GFP dataset/environment code, except its D4RL init-order typo."""
    benchmark = _benchmark(config.env)
    if benchmark != "D4RL":
        return (*original_make_env_and_datasets(
            benchmark, config.env,
            frame_stack=config.frame_stack,
            action_clip_eps=config.action_clip_eps,
        ), benchmark)

    # The uploaded original env_utils.py attempts get_dataset(env,...) before
    # creating `env`; correct only that I/O initialization order, keeping its
    # ORIGINAL d4rl_utils.get_dataset() and .make_env() implementations.
    from algorithms.uk_offline.gfp_official.envs import d4rl_utils
    env = d4rl_utils.make_env(config.env)
    train_dataset = d4rl_utils.get_dataset(env, config.env)
    if config.action_clip_eps is not None:
        eps = float(config.action_clip_eps)
        train_dataset = train_dataset.copy(add_or_replace={
            "actions": np.clip(train_dataset["actions"], -1 + eps, 1 - eps)
        })
    def new_eval_env():
        eval_env = d4rl_utils.make_env(config.env)
        eval_env.reset()
        return eval_env
    env.reset()
    return env, new_eval_env, train_dataset, None, benchmark


def _safe_float(x: Any) -> float:
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return float("nan")


def evaluate_policy(agent_fn, eval_env: Any, episodes: int, seed: int, benchmark: str):
    """FQL/ORL-style per-episode logging; same GFP.sample_actions/flow_actions."""
    rng = jax.random.PRNGKey(int(seed))
    stats: Dict[str, list] = {}
    returns: list[float] = []
    successes: list[float] = []
    for ep in trange(episodes, desc="GFP Eval", leave=False):
        try:
            observation, info = eval_env.reset(seed=seed if ep == 0 else None)
        except TypeError:
            observation, info = eval_env.reset()
        done = False
        total_reward = 0.0
        last_info: Dict[str, Any] = {}
        while not done:
            rng, action_rng = jax.random.split(rng)
            action = np.asarray(agent_fn(observations=observation, seed=action_rng, temperature=0))
            action = np.clip(action, -1, 1)
            observation, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
            last_info = dict(info)
        # Same success definition as the official evaluate and EpisodeMonitor.
        def flatten(d, prefix=""):
            for k, v in d.items():
                key = prefix + str(k)
                if hasattr(v, "items"):
                    yield from flatten(v, key + ".")
                else:
                    yield key, v
        flat = dict(flatten(last_info))
        for k, v in flat.items():
            fval = _safe_float(v)
            if np.isfinite(fval):
                stats.setdefault(k, []).append(fval)
        returns.append(float(flat.get("episode.return", total_reward)))
        suc = flat.get("success", flat.get("episode.success", float("nan")))
        successes.append(_safe_float(suc))
    arr_rew = np.asarray(returns, dtype=np.float32)
    arr_suc = np.asarray(successes, dtype=np.float32)
    output = {k: float(np.mean(vals)) for k, vals in stats.items()}
    output["reward_mean"] = float(np.mean(arr_rew))
    output["reward_std"] = float(np.std(arr_rew))
    valid = arr_suc[np.isfinite(arr_suc)]
    output["success_rate"] = float(np.mean(valid)) if valid.size else float("nan")
    output["success_std"] = float(np.std(valid)) if valid.size else float("nan")
    if benchmark == "D4RL":
        raw = np.asarray([float(v) for v in stats.get("episode.normalized_return", [])], dtype=np.float32)
        output["d4rl_normalized_score_mean"] = float(np.mean(raw)) if raw.size else float("nan")
        output["d4rl_normalized_score_std"] = float(np.std(raw)) if raw.size else float("nan")
    elif benchmark == "Minari":
        from algorithms.uk_offline.gfp_official.envs.minari_utils import minari_normalized_score
        scores = np.asarray([minari_normalized_score(v) for v in arr_rew], dtype=np.float32)
        output["d4rl_normalized_score_mean"] = float(np.mean(scores))
        output["d4rl_normalized_score_std"] = float(np.std(scores))
    else:
        output["d4rl_normalized_score_mean"] = float("nan")
        output["d4rl_normalized_score_std"] = float("nan")
    return output


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_pickle(path: Union[Path, str]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_checkpoint_path(load_model: str, name: str, seed: int) -> Path:
    path = Path(load_model)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"--load_model not found: {path}")
    for filename in ("checkpoint.pkl", "best_checkpoint.pkl"):
        for candidate in (path / filename, path / name / str(seed) / filename):
            if candidate.is_file():
                return candidate
    results = [c for pattern in ("*/checkpoint.pkl", "*/best_checkpoint.pkl", "*/*/checkpoint.pkl", "*/*/best_checkpoint.pkl") for c in sorted(path.glob(pattern))]
    results = list(dict.fromkeys(results))
    if len(results) == 1:
        return results[0]
    if not results:
        raise FileNotFoundError(f"Could not find checkpoint.pkl/best_checkpoint.pkl under {path}")
    raise ValueError("Multiple checkpoint candidates found; specify an exact path: " + ", ".join(map(str, results)))


def save_eval_logs(eval_logs: List[Dict[str, Any]], path: Optional[str]) -> None:
    if not path or not eval_logs:
        return
    keys = list(dict.fromkeys(k for row in eval_logs for k in row))
    arrays = {}
    for key in keys:
        vals = [row.get(key, float("nan")) for row in eval_logs]
        try:
            arrays[key] = np.asarray(vals, dtype=np.float64)
        except (ValueError, TypeError):
            arrays[key] = np.asarray(vals, dtype=object)
    np.savez(os.path.join(path, "eval_logs.npz"), **arrays)


def _make_payload(agent, cfg: TrainConfig, step: int):
    return {
        "agent": serialization.to_state_dict(agent),
        "config": asdict(cfg),
        "_training_timestep": int(step),
        "_numpy_rng_state": np.random.get_state(),
        "_python_rng_state": random.getstate(),
    }


def _restore_payload(agent, payload):
    state = payload["agent"] if isinstance(payload, dict) and "agent" in payload else payload
    agent = serialization.from_state_dict(agent, state)
    if isinstance(payload, dict):
        if "_numpy_rng_state" in payload:
            np.random.set_state(payload["_numpy_rng_state"])
        if "_python_rng_state" in payload:
            random.setstate(payload["_python_rng_state"])
    return agent


def _wandb_init(cfg: TrainConfig, manager) -> bool:
    if not cfg.log_wandb:
        return False
    if manager is not None:
        ok = manager.initialize_wandb_with_fallback(
            wandb_module=wandb, config=asdict(cfg), code_root=_PROJECT_ROOT,
        )
        if not ok:
            cfg.log_wandb = False
        return bool(ok)
    try:
        run = wandb.init(
            config=asdict(cfg), project=cfg.project, group=cfg.group,
            name=cfg.name, id=str(uuid.uuid4()),
        )
        if run is not None and hasattr(run, "log_code"):
            run.log_code(str(_PROJECT_ROOT))
        return True
    except Exception as exc:
        print(f"Warning: W&B initialization failed: {exc}; continuing locally")
        cfg.log_wandb = False
        return False


def _train_impl(config: TrainConfig):
    config = finalize_checkpoint_path(apply_env_hyperparams(config))
    run_dir = Path(config.checkpoints_path) if config.checkpoints_path else None
    manager, preparation = None, None
    if run_dir is not None:
        manager = TrainingCheckpointManager(
            run_dir=str(run_dir),
            current_config=asdict(config),
            default_config=asdict(TrainConfig()),
            max_timesteps=int(config.max_timesteps),
            checkpoint_type="gfp_jax_training_progress",
            identity_ignored_fields=DEFAULT_IDENTITY_IGNORED_FIELDS,
            checkpoint_version=2,
            accepted_checkpoint_versions=(1, 2),
            wandb_enabled=bool(config.log_wandb),
            wandb_entity=config.wandb_entity,
            wandb_project=config.project,
            final_checkpoint_name="checkpoint.pkl",
        )
        preparation = manager.prepare()
        print(preparation.message)
        if preparation.is_completed:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        if not (run_dir / "config.yaml").exists():
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(asdict(config), f, sort_keys=True, allow_unicode=True)

    dev = select_jax_device(config.device)
    train_env, eval_env_creator, train_dataset, val_dataset, benchmark = make_env_and_datasets(config)
    # Match original offline GFP's replay-buffer sampling. For visual data, do
    # not allocate the unnecessary, potentially enormous 2M image replay buffer.
    if not config.env.startswith("visual-"):
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(config.buffer_size, train_dataset.size + 1)
        )
    for dataset in (train_dataset, val_dataset):
        if dataset is not None:
            dataset.p_aug = config.p_aug
            dataset.frame_stack = config.frame_stack
            dataset.return_next_actions = False

    set_seed(config.seed, train_env)
    example_batch = train_dataset.sample(1)
    original_cfg = agent_config(config)
    agent = GFPAgent.create(config.seed, example_batch["observations"], example_batch["actions"], original_cfg)
    agent = jax.device_put(agent, dev)
    print("-" * 60)
    print(f"Training GFP-JAX, Env: {config.env}, Seed: {config.seed}")
    print(f"Original GFP agent config: {original_cfg}")
    print(f"Saving under: {run_dir if run_dir else '(disabled)'}")
    print("-" * 60)

    train_step = 0
    eval_logs: List[Dict[str, Any]] = []
    if config.load_model:
        file = _resolve_checkpoint_path(config.load_model, config.name, config.seed)
        payload = _load_pickle(file)
        agent = jax.device_put(_restore_payload(agent, payload), dev)
        train_step = int(payload.get("_training_timestep", 0)) if isinstance(payload, dict) else 0
        print(f"Loaded GFP checkpoint {file} at step {train_step}")
        if run_dir is not None and file.parent == run_dir and (run_dir / "eval_logs.npz").is_file():
            with np.load(run_dir / "eval_logs.npz", allow_pickle=True) as z:
                keys = list(z.files)
                count = len(z[keys[0]]) if keys else 0
                eval_logs = [dict((k, z[k][i].item() if np.asarray(z[k][i]).ndim == 0 else z[k][i]) for k in keys) for i in range(count)]

    def current_step():
        return int(train_step)

    def progress_state():
        return _make_payload(agent, config, train_step)

    def load_progress(payload):
        nonlocal agent, train_step
        agent = jax.device_put(_restore_payload(agent, payload), dev)
        train_step = int(payload.get("_training_timestep", 0))

    if manager is not None:
        if preparation.is_resuming:
            train_step, eval_logs, _ = manager.restore(
                load_trainer_state=load_progress, get_restored_timestep=current_step,
            )
            print(f"Automatically restored GFP training at timestep {train_step}")
        else:
            manager.save_progress(
                timestep=train_step, trainer_state=progress_state(),
                eval_logs=eval_logs, status="running",
            )

    # W&B initialization is I/O, so keep its RNG use from altering training draws.
    _before_wandb_np = np.random.get_state()
    _before_wandb_py = random.getstate()
    _wandb_init(config, manager)
    np.random.set_state(_before_wandb_np)
    random.setstate(_before_wandb_py)

    def wandb_log(metrics: Dict[str, float], step: int):
        if not config.log_wandb or not metrics:
            return
        if manager is not None:
            manager.log_wandb(metrics, int(step))
        else:
            wandb.log(metrics, step=int(step))

    # The official GPU fused path samples directly with the JAX RNG from a
    # device-resident dataset; the exact official multi_sample_and_update runs.
    data_gpu = None
    if config.dataset_on_gpu:
        data_gpu = {k: jax.device_put(np.asarray(v), dev) for k, v in train_dataset.items()}
        dataset_size = int(train_dataset.size)

    first_time = time.time()
    prev_time = first_time
    last_update_info: Dict[str, Any] = {}
    best_eval_metric_mean = best_eval_metric(eval_logs)
    first_eval_step = config.n_steps_fused if config.dataset_on_gpu else 1

    def evaluation_is_required(step: int) -> bool:
        return (
            (config.eval_at_first_step and step == first_eval_step)
            or step % int(config.eval_freq) == 0
            or step == int(config.max_timesteps)
        )

    try:
        for step in trange(train_step + 1, int(config.max_timesteps) + 1, desc="GFP Training"):
            if config.dataset_on_gpu and step % config.n_steps_fused:
                continue
            if config.dataset_on_gpu:
                agent, update_info = agent.multi_sample_and_update(
                    int(config.n_steps_fused), data_gpu, dataset_size, int(config.batch_size)
                )
            else:
                batch = train_dataset.sample(int(config.batch_size))
                agent, update_info = agent.update(batch)
            train_step = int(step)
            last_update_info = {k: float(jax.device_get(v)) for k, v in update_info.items()}

            if config.log_wandb and config.log_metrics and train_step % int(config.log_every) == 0:
                train_info = {f"training/{key}": val for key, val in last_update_info.items()}
                if val_dataset is not None:
                    val_batch = val_dataset.sample(int(config.batch_size))
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                    train_info.update({
                        f"validation/{k}": float(jax.device_get(v))
                        for k, v in val_info.items()
                    })
                train_info["time/epoch_time"] = (time.time() - prev_time) / int(config.log_every)
                train_info["time/total_time"] = time.time() - first_time
                prev_time = time.time()
                wandb_log(train_info, train_step)

            if evaluation_is_required(train_step):
                actor_env = eval_env_creator()
                actor_results = evaluate_policy(
                    agent.sample_actions, actor_env, int(config.n_episodes),
                    int(config.eval_seed), benchmark,
                )
                if hasattr(actor_env, "close"):
                    actor_env.close()
                eval_log: Dict[str, Any] = {
                    "timestep": int(train_step),
                    "eval/reward_mean": actor_results["reward_mean"],
                    "eval/reward_std": actor_results["reward_std"],
                    "eval/d4rl_normalized_score_mean": actor_results["d4rl_normalized_score_mean"],
                    "eval/d4rl_normalized_score_std": actor_results["d4rl_normalized_score_std"],
                    "eval/success_rate": actor_results["success_rate"],
                    "eval/success_std": actor_results["success_std"],
                }
                if config.eval_flow_policy:
                    flow_env = eval_env_creator()
                    flow_results = evaluate_policy(
                        agent.sample_flow_actions, flow_env, int(config.n_episodes),
                        int(config.eval_seed), benchmark,
                    )
                    if hasattr(flow_env, "close"):
                        flow_env.close()
                    for key in ("reward_mean", "reward_std", "success_rate", "success_std"):
                        eval_log["eval_flow/" + key] = flow_results[key]
                    eval_log["eval_flow/d4rl_normalized_score_mean"] = flow_results["d4rl_normalized_score_mean"]
                    eval_log["eval_flow/d4rl_normalized_score_std"] = flow_results["d4rl_normalized_score_std"]
                if benchmark == "OGBench" and train_step == int(config.max_timesteps):
                    previous = [
                        log.get("eval/success_rate", float("nan"))
                        for log in eval_logs
                        if log.get("timestep", -1) < train_step
                    ]
                    values = previous[-2:] + [eval_log["eval/success_rate"]]
                    if len(values) == 3 and np.all(np.isfinite(values)):
                        # Official GFP paper reports the last three evaluations.
                        eval_log["eval/final_last3_success_rate"] = float(np.mean(values))
                        eval_log["eval/final_last3_success_score"] = float(100 * np.mean(values))
                upsert_eval_log(eval_logs, eval_log)
                print(f"Step {train_step} GFP Actor: reward={eval_log['eval/reward_mean']:.3f} ± "
                      f"{eval_log['eval/reward_std']:.3f}, success={eval_log['eval/success_rate']:.3f} ± "
                      f"{eval_log['eval/success_std']:.3f}")
                if config.eval_flow_policy:
                    print(f"    GFP VaBC flow: success={eval_log['eval_flow/success_rate']:.3f}")
                wandb_log(eval_log, train_step)
                save_eval_logs(eval_logs, config.checkpoints_path)

                if config.save_best_model and run_dir is not None:
                    # ORL convention: select actor only by success, then normalized
                    # score, then reward. Flow policy is auxiliary, never used for best.
                    metric = eval_log["eval/success_rate"]
                    if not np.isfinite(metric):
                        metric = eval_log["eval/d4rl_normalized_score_mean"]
                    if not np.isfinite(metric):
                        metric = eval_log["eval/reward_mean"]
                    if np.isfinite(metric) and metric > best_eval_metric_mean:
                        best_eval_metric_mean = metric
                        _save_pickle(run_dir / "best_checkpoint.pkl", progress_state())

            if manager is not None and train_step % int(config.checkpoint_freq) == 0:
                manager.save_progress(
                    timestep=train_step, trainer_state=progress_state(),
                    eval_logs=eval_logs, status="running",
                )
    except BaseException:
        if manager is not None:
            committed = (
                not evaluation_is_required(int(train_step))
                or find_eval_log(eval_logs, int(train_step)) is not None
            )
            if committed:
                manager.save_progress(
                    timestep=int(train_step), trainer_state=progress_state(),
                    eval_logs=eval_logs, status="interrupted",
                )
                print(f"Saved interrupted GFP training at timestep {train_step}")
            else:
                print("GFP evaluation was interrupted; retaining previous safe checkpoint")
        raise

    if manager is not None:
        path = manager.complete(
            timestep=int(train_step), final_state=progress_state(),
            save_final_model=bool(config.save_final_model), eval_logs=eval_logs,
        )
        if path is not None:
            print(f"Saved final GFP checkpoint: {path}")
    if hasattr(train_env, "close"):
        train_env.close()


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
        if getattr(wandb, "run", None) is not None:
            wandb.finish(exit_code=exit_code)


if __name__ == "__main__":
    train()
