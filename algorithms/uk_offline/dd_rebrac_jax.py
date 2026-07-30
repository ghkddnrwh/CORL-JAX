import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import collections
import glob
import pickle
from functools import partial
import random
import re
import sys
import time
import uuid
from copy import deepcopy
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
from tqdm.auto import trange

try:
    import gymnasium
    from gymnasium.spaces import Box
except ImportError:
    gymnasium = None
    Box = None

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

TensorBatch = Dict[str, jnp.ndarray]

ALGORITHM_NAME = "DD-ReBRAC"
ALGORITHM_FULL_NAME = "Delayed-Policy Ensemble-Average ReBRAC"


@dataclass
class TrainConfig:
    device: str = "gpu"
    env: str = "halfcheetah-medium-v2"
    seed: int = 0
    eval_seed: int = 42

    max_timesteps: int = int(1e6)
    eval_freq: int = int(100000)
    n_episodes: int = 50

    checkpoints_path: Optional[str] = None
    load_model: str = ""
    mode: str = "train"
    hyperparams_path: Optional[str] = "hyperparams/dd_rebrac_jax.yml"  # Reuses ReBRAC hyperparams unless a DD-ReBRAC YAML is supplied.
    use_hyperparams: bool = True
    dataset_name: Optional[str] = None

    batch_size: int = 256
    normalize_reward: bool = False
    normalize_states: bool = False
    action_clip_eps: Optional[float] = 1e-5
    frame_stack: Optional[int] = None
    p_aug: Optional[float] = None

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    hidden_dim: int = 512
    actor_n_hiddens: int = 4
    critic_n_hiddens: int = 4
    num_critics: int = 2
    discount: float = 0.99
    tau: float = 5e-3
    actor_bc_coef: float = 0.0
    critic_bc_coef: float = 0.0
    actor_ln: bool = False
    critic_ln: bool = True
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 1
    delayed_update_period: int = 250
    normalize_q: bool = True
    tanh_squash: bool = True
    actor_fc_scale: float = 0.01
    encoder: Optional[str] = None

    actor_hidden_dims: Optional[Tuple[int, ...]] = None
    value_hidden_dims: Optional[Tuple[int, ...]] = None

    actor_refit_dir_name: str = "actor_refit"

    project: str = "ORL-SMOOTH"
    group: str = "ReBRAC-JAX"
    name: str = "ReBRAC-JAX"
    log_wandb: bool = True
    log_every: int = 500
    save_final_model: bool = False
    save_best_model: bool = False
    eval_at_first_step: bool = True

    checkpoint_freq: int = int(25e3)
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


def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = "ORL-BIAS"
    config.group = f"{ALGORITHM_NAME}-JAX-FQL"
    config.name = f"{ALGORITHM_NAME}-JAX-FQL-{config.env}"


def validate_config(config: TrainConfig) -> None:
    assert config.mode in ("train", "actor_refit"), "mode must be train or actor_refit"
    assert config.batch_size > 0
    assert config.eval_freq > 0
    assert config.n_episodes > 0
    assert config.max_timesteps >= 0
    assert config.log_every > 0
    assert config.checkpoint_freq > 0
    assert config.num_critics > 0
    assert config.discount >= 0.0 and config.discount <= 1.0
    assert config.tau >= 0.0 and config.tau <= 1.0
    assert config.actor_bc_coef >= 0.0
    assert config.critic_bc_coef >= 0.0
    assert config.policy_noise >= 0.0
    assert config.noise_clip >= 0.0
    assert config.policy_freq > 0
    assert config.delayed_update_period > 0
    assert config.actor_refit_dir_name != ""
    if config.mode == "actor_refit":
        assert config.load_model != "", "mode='actor_refit' requires --load_model"
    if config.actor_learning_rate != config.critic_learning_rate:
        print(
            "Warning: FQL ReBRAC uses one shared Adam optimizer for actor and critic. "
            f"Using actor_learning_rate={config.actor_learning_rate}; "
            f"critic_learning_rate={config.critic_learning_rate} is ignored."
        )
    if config.num_critics < 2:
        print(
            "Warning: DD-ReBRAC is intended to use an ensemble of critics. "
            f"This run uses num_critics={config.num_critics}."
        )


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
        "n_timesteps": "max_timesteps",
        "lr": "actor_learning_rate",
        "alpha_actor": "actor_bc_coef",
        "alpha_critic": "critic_bc_coef",
        "actor_freq": "policy_freq",
        "actor_noise": "policy_noise",
        "actor_noise_clip": "noise_clip",
        "delay_update_period": "delayed_update_period",
        "delayed_update_interval": "delayed_update_period",
        "delay_policy_update_period": "delayed_update_period",
        "layer_norm": "critic_ln",
        "actor_layer_norm": "actor_ln",
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
        if raw_key == "lr" and "critic_learning_rate" not in cli_overrides:
            config.critic_learning_rate = _coerce_hparam_value(raw_value)
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
        print(f"Ignored unknown hyperparameter keys for ReBRAC-FQL: {', '.join(skipped_unknown)}")
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
            if hasattr(env, "seed"):
                env.seed(seed)
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


def save_and_upload_eval_logs(eval_logs: List[Dict[str, Any]], checkpoints_path: Optional[str], log_wandb: bool):
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


def mean_std_or_nan(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan, np.nan
    return float(np.mean(finite_values)), float(np.std(finite_values))


def get_size(data):
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@jax.jit
def _identity_jit(x):
    return x


def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    return nn.vmap(
        cls,
        variable_axes={"params": 0, "intermediates": 0},
        split_rngs={"params": True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


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
            if i == len(self.hidden_dims) - 2:
                self.sow("intermediates", "feature", x)
        return x


class TransformedWithMode(distrax.Transformed):
    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

    def __call__(self, observations, temperature=1.0):
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))
        return distribution


class Value(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

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
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding="SAME",
                kernel_init=initializer,
            )(conv_out)
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding="SAME",
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input
        return conv_out


class ImpalaEncoder(nn.Module):
    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
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
    "impala_debug": lambda: ImpalaEncoder(num_blocks=1, stack_sizes=(4, 4)),
    "impala_small": lambda: ImpalaEncoder(num_blocks=1),
    "impala_large": lambda: ImpalaEncoder(stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
}


class ModuleDict(nn.Module):
    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f"When `name` is not specified, kwargs must contain exactly module keys. "
                    f"Got {kwargs.keys()} and expected {self.modules.keys()}."
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out
        return self.modules[name](*args, **kwargs)


nonpytree_field = flax.struct.field


class FQLTrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Any = flax.struct.field(pytree_node=False)
    model_def: Any = flax.struct.field(pytree_node=False)
    params: Any
    tx: Any = flax.struct.field(pytree_node=False)
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        params = unfreeze(params)
        opt_state = tx.init(params) if tx is not None else None
        return cls(step=1, apply_fn=model_def.apply, model_def=model_def, params=params, tx=tx, opt_state=opt_state, **kwargs)

    def __call__(self, *args, params=None, method=None, **kwargs):
        if params is None:
            params = self.params
        variables = {"params": params}
        method_name = getattr(self.model_def, method) if method is not None else None
        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        return lambda *args, params=None, **kwargs: self(*args, params=params, name=name, **kwargs)

    def apply_gradients(self, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)

    def apply_loss_fn(self, loss_fn):
        # Keep the FQL/ReBRAC algorithmic update intact, but do not compute
        # global gradient diagnostics inside the jitted update step. Those
        # diagnostics create large reduce operations over every parameter leaf
        # and can trigger very slow XLA constant-folding during first compile.
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
        return self.apply_gradients(grads=grads), info


class DDReBRACAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = flax.struct.field(pytree_node=False)

    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        # DD-ReBRAC: the policy used inside the Bellman target is a hard-delayed
        # copy of the online actor, not a soft target actor.
        next_dist = self.network.select("delayed_actor")(batch["next_observations"])
        next_actions = next_dist.mode()
        noise = jnp.clip(
            jax.random.normal(sample_rng, next_actions.shape) * self.config["actor_noise"],
            -self.config["actor_noise_clip"],
            self.config["actor_noise_clip"],
        )
        next_actions = jnp.clip(next_actions + noise, -1, 1)

        # DD-ReBRAC: replace clipped double-Q/min target by the ensemble mean of
        # all target critics evaluated at the delayed-policy action.
        next_qs = self.network.select("target_critic")(batch["next_observations"], actions=next_actions)
        next_q = next_qs.mean(axis=0)
        mse = jnp.square(next_actions - batch["next_actions"]).sum(axis=-1)
        next_q = next_q - self.config["alpha_critic"] * mse
        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q
        q = self.network.select("critic")(batch["observations"], actions=batch["actions"], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "target_q_mean": target_q.mean(),
            "target_q_std": target_q.std(),
            "target_critic_mean": next_qs.mean(),
            "target_critic_std": next_qs.std(),
        }

    def actor_loss(self, batch, grad_params, rng):
        dist = self.network.select("actor")(batch["observations"], params=grad_params)
        actions = dist.mode()
        qs = self.network.select("critic")(batch["observations"], actions=actions)
        q = qs.mean(axis=0)
        mse = jnp.square(actions - batch["actions"]).sum(axis=-1)
        if self.config.get("normalize_q", True):
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        else:
            lam = 1.0
        actor_loss = -(lam * q).mean()
        bc_loss = (self.config["alpha_actor"] * mse).mean()
        total_loss = actor_loss + bc_loss
        if self.config["tanh_squash"]:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev().mean()
        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "bc_loss": bc_loss,
            "std": action_std.mean(),
            "mse": mse.mean(),
            "q_mean": q.mean(),
            "q_std": qs.std(),
        }

    @jax.jit
    def total_loss_eval(self, batch, rng=None):
        return self.total_loss(batch, grad_params=None, full_update=True, rng=rng)

    def total_loss(self, batch, grad_params, full_update=True, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v
        if full_update:
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f"actor/{k}"] = v
        else:
            actor_loss = 0.0
        loss = critic_loss + actor_loss
        return loss, info

    def _target_update(self, network, module_name):
        source_key = f"modules_{module_name}"
        target_key = f"modules_target_{module_name}"
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            network.params[source_key],
            network.params[target_key],
        )
        new_params = dict(network.params)
        new_params[target_key] = new_target_params
        return network.replace(params=new_params)

    def _delayed_actor_update(self, network, total_it):
        should_update = (total_it % jnp.asarray(self.config["delayed_update_period"], dtype=jnp.int32)) == 0
        new_delayed_params = jax.lax.cond(
            should_update,
            lambda _: network.params["modules_actor"],
            lambda _: network.params["modules_delayed_actor"],
            operand=None,
        )
        new_params = dict(network.params)
        new_params["modules_delayed_actor"] = new_delayed_params
        return network.replace(params=new_params), should_update

    @jax.jit
    def update_critic_only(self, batch):
        return self.update(batch, full_update=False)

    @jax.jit
    def update_full(self, batch):
        return self.update(batch, full_update=True)

    def update(self, batch, full_update=True):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if full_update:
            new_network = self._target_update(new_network, "critic")

        # Hard update delayed_actor every d optimization steps. This is deliberately
        # independent of the soft target critic update and replaces target_actor.
        optimization_step = new_network.step - 1
        new_network, delayed_update = self._delayed_actor_update(new_network, optimization_step)
        info["delayed_actor/update"] = delayed_update.astype(jnp.float32)
        info["delayed_actor/update_period"] = jnp.asarray(self.config["delayed_update_period"], dtype=jnp.float32)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        dist = self.network.select("actor")(observations, temperature=temperature)
        actions = dist.mode()
        noise = jnp.clip(
            jax.random.normal(seed, actions.shape) * self.config["actor_noise"] * temperature,
            -self.config["actor_noise_clip"],
            self.config["actor_noise_clip"],
        )
        return jnp.clip(actions + noise, -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        action_dim = ex_actions.shape[-1]
        encoders = {}
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor"] = encoder_module()
        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config.get("num_critics", 2),
            encoder=encoders.get("critic"),
        )
        actor_def = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=config["tanh_squash"],
            state_dependent_std=False,
            const_std=True,
            final_fc_init_scale=config["actor_fc_scale"],
            encoder=encoders.get("actor"),
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
            delayed_actor=(deepcopy(actor_def), (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = FQLTrainState.create(network_def, network_params, tx=network_tx)
        params = dict(network.params)
        params["modules_target_critic"] = deepcopy(params["modules_critic"])
        params["modules_delayed_actor"] = deepcopy(params["modules_actor"])
        network = network.replace(params=params, opt_state=network_tx.init(params))
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


class FQLDataset(FrozenDict):
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
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1]) if len(self.terminal_locs) > 0 else np.array([0])

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
            batch["observations"] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch["next_observations"] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None:
            if np.random.rand() < self.p_aug:
                self.augment(batch, ["observations", "next_observations"])
        return batch

    def get_subset(self, idxs):
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        result = dict(result)
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


@partial(jax.jit, static_argnames=("padding",))
def random_crop(img, crop_from, padding):
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(imgs, crop_froms, padding):
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class ReplayBuffer:
    def __init__(self, device: Any):
        self.dataset: Optional[FQLDataset] = None
        self.mean: Union[np.ndarray, float] = 0.0
        self.std: Union[np.ndarray, float] = 1.0
        self.device = device

    def create_from_dataset(
        self,
        env_name: str,
        dataset: FQLDataset,
        normalize_reward: bool = False,
        is_normalize: bool = False,
        p_aug: Optional[float] = None,
        frame_stack: Optional[int] = None,
    ):
        data = dict(dataset)
        if "masks" not in data:
            if "terminals" in data:
                data["masks"] = 1.0 - np.asarray(data["terminals"], dtype=np.float32)
            else:
                data["masks"] = np.ones_like(np.asarray(data["rewards"], dtype=np.float32))
        if normalize_reward:
            data["rewards"] = self.normalize_reward(env_name, np.asarray(data["rewards"], dtype=np.float32))
        if is_normalize:
            self.mean, self.std = compute_mean_std(np.asarray(data["observations"], dtype=np.float32), eps=1e-3)
            data["observations"] = normalize_states(data["observations"], self.mean, self.std)
            data["next_observations"] = normalize_states(data["next_observations"], self.mean, self.std)
        data = {k: (np.asarray(v) if np.asarray(v).dtype == np.uint8 else np.asarray(v, dtype=np.float32)) for k, v in data.items()}
        self.dataset = FQLDataset.create(**data)
        self.dataset.return_next_actions = True
        self.dataset.p_aug = p_aug
        self.dataset.frame_stack = frame_stack
        print(f"Dataset size: {self.size}")

    @property
    def size(self) -> int:
        return 0 if self.dataset is None else int(self.dataset.size)

    @property
    def state_dim(self) -> int:
        obs = self.dataset["observations"]
        if isinstance(obs, dict):
            raise ValueError("Dict observations are not supported by this standalone runner.")
        if self.dataset.frame_stack is not None:
            return int(obs.shape[-1] * self.dataset.frame_stack)
        return int(obs.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def sample_batch(self, key: Optional[jax.random.PRNGKey], batch_size: int) -> TensorBatch:
        del key
        batch = self.dataset.sample(batch_size)
        return tree_to_device({k: jnp.asarray(v) for k, v in batch.items()}, self.device)

    @staticmethod
    def normalize_reward(dataset_name: str, rewards: np.ndarray) -> np.ndarray:
        if "antmaze" in dataset_name:
            return rewards * 100.0
        raise NotImplementedError("Reward normalization is implemented only for AntMaze.")


def compute_mean_std(states: Union[np.ndarray, jax.Array], eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(states).mean(0)
    std = np.asarray(states).std(0) + eps
    return mean, std


def normalize_states(states: Union[np.ndarray, jax.Array], mean: Union[np.ndarray, float], std: Union[np.ndarray, float]) -> np.ndarray:
    return (states - mean) / std


class TransformEnv:
    def __init__(self, env: Any, state_mean: Union[np.ndarray, float], state_std: Union[np.ndarray, float]):
        self.env = env
        self.state_mean = state_mean
        self.state_std = state_std
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def _normalize_state(self, state):
        return (state - self.state_mean) / self.state_std

    def reset(self, *args, **kwargs):
        state, info = self.env.reset(*args, **kwargs)
        return self._normalize_state(state), info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize_state(state), reward, terminated, truncated, info


class EpisodeMonitor(gymnasium.Wrapper if gymnasium is not None else object):
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


class FrameStackWrapper(gymnasium.Wrapper if gymnasium is not None else object):
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


def make_d4rl_env(env_name):
    if gymnasium is None:
        raise ImportError("gymnasium is required for D4RL envs in this runner.")
    env = gymnasium.make("GymV21Environment-v0", env_id=env_name)
    return EpisodeMonitor(env)


def get_d4rl_dataset(env, env_name):
    global d4rl
    if d4rl is None:
        try:
            import d4rl as d4rl_module
        except Exception as exc:
            raise ImportError("D4RL environment requested, but d4rl could not be imported.") from exc
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
    return FQLDataset.create(
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
        train_dataset = FQLDataset.create(**train_dataset)
        val_dataset = FQLDataset.create(**val_dataset)
    elif "antmaze" in env_name and ("diverse" in env_name or "play" in env_name or "umaze" in env_name):
        env = make_d4rl_env(env_name)
        eval_env = make_d4rl_env(env_name)
        train_dataset = get_d4rl_dataset(env, env_name)
        val_dataset = None
    elif "pen" in env_name or "hammer" in env_name or "relocate" in env_name or "door" in env_name:
        env = make_d4rl_env(env_name)
        eval_env = make_d4rl_env(env_name)
        train_dataset = get_d4rl_dataset(env, env_name)
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


def evaluate(agent, env, config=None, num_eval_episodes=50, eval_temperature=0.0):
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    stats = collections.defaultdict(list)
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
        add_to(stats, flatten(info))
    for k, v in stats.items():
        stats[k] = np.mean(v)
    return dict(stats)


def fql_config_from_train_config(config: TrainConfig) -> Dict[str, Any]:
    actor_hidden_dims = config.actor_hidden_dims or tuple([config.hidden_dim] * config.actor_n_hiddens)
    value_hidden_dims = config.value_hidden_dims or tuple([config.hidden_dim] * config.critic_n_hiddens)
    return {
        "agent_name": "dd_rebrac",
        "lr": config.actor_learning_rate,
        "batch_size": config.batch_size,
        "actor_hidden_dims": actor_hidden_dims,
        "value_hidden_dims": value_hidden_dims,
        "layer_norm": config.critic_ln,
        "actor_layer_norm": config.actor_ln,
        "discount": config.discount,
        "tau": config.tau,
        "tanh_squash": config.tanh_squash,
        "actor_fc_scale": config.actor_fc_scale,
        "alpha_actor": config.actor_bc_coef,
        "alpha_critic": config.critic_bc_coef,
        "actor_freq": config.policy_freq,
        "actor_noise": config.policy_noise,
        "actor_noise_clip": config.noise_clip,
        "encoder": config.encoder,
        "normalize_q": config.normalize_q,
        "num_critics": config.num_critics,
        "delayed_update_period": config.delayed_update_period,
    }


@struct.dataclass
class ActorRefitState:
    params: Any
    opt_state: Any
    key: jnp.ndarray


class DDReBRACFQLJAX:
    def __init__(self, config: TrainConfig, ex_observations: np.ndarray, ex_actions: np.ndarray, device: Any = None):
        self.config = config
        self.fql_config = fql_config_from_train_config(config)
        self.device = device if device is not None else jax.devices()[0]
        self.agent = DDReBRACAgent.create(config.seed, jnp.asarray(ex_observations), jnp.asarray(ex_actions), self.fql_config)
        self.agent = tree_to_device(self.agent, self.device)
        self.total_it = 0
        self.actor_tx = optax.adam(learning_rate=self.fql_config["lr"])
        self.initial_actor_params = deepcopy(self.agent.network.params["modules_actor"])
        actor_refit_key = jax.random.PRNGKey(config.seed + 1)
        self.initial_actor_key = tree_to_device(actor_refit_key, self.device)
        self.initial_actor_opt_state = tree_to_device(self.actor_tx.init(self.initial_actor_params), self.device)
        self._actor_refit_step = self._build_actor_refit_step()

    def train(self, batch: TensorBatch, update_actor_now: bool) -> Dict[str, float]:
        if update_actor_now:
            self.agent, log_dict = self.agent.update_full(batch)
        else:
            self.agent, log_dict = self.agent.update_critic_only(batch)
        self.total_it += 1
        return {key: float(jax.device_get(value)) for key, value in log_dict.items()}

    def eval_current_actor(self, env: Any, n_episodes: int) -> Dict[str, float]:
        return evaluate(self.agent, env, config=self.fql_config, num_eval_episodes=n_episodes, eval_temperature=0.0)

    def _actor_full_params(self, actor_params):
        full_params = dict(self.agent.network.params)
        full_params["modules_actor"] = actor_params
        return full_params

    def actor_act(self, actor_params: Any, state: np.ndarray) -> np.ndarray:
        state_jnp = tree_to_device(jnp.asarray(state.reshape(1, *state.shape), dtype=jnp.float32), self.device)
        full_params = self._actor_full_params(actor_params)
        dist = self.agent.network.select("actor")(state_jnp, params=full_params, temperature=0.0)
        action = dist.mode()
        return np.asarray(jax.device_get(action))[0]

    def eval_actor(self, env: Any, actor_params: Any, n_episodes: int) -> Dict[str, float]:
        stats = collections.defaultdict(list)
        for _ in trange(n_episodes, desc="Eval", leave=False):
            observation, info = env.reset()
            done = False
            while not done:
                action = self.actor_act(actor_params, observation)
                action = np.clip(action, -1, 1)
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            add_to(stats, flatten(info))
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def _build_actor_refit_step(self):
        actor_tx = self.actor_tx
        alpha_actor = self.fql_config["alpha_actor"]
        normalize_q = self.fql_config.get("normalize_q", True)
        network = self.agent.network

        @jax.jit
        def actor_refit_step(actor_state: ActorRefitState, agent: DDReBRACAgent, batch: TensorBatch):
            key, _ = jax.random.split(actor_state.key)

            def actor_loss_fn(actor_params):
                full_params = dict(agent.network.params)
                full_params["modules_actor"] = actor_params
                dist = agent.network.select("actor")(batch["observations"], params=full_params)
                actions = dist.mode()
                qs = agent.network.select("critic")(batch["observations"], actions=actions)
                q = qs.mean(axis=0)
                mse = jnp.square(actions - batch["actions"]).sum(axis=-1)
                lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean()) if normalize_q else 1.0
                actor_loss = -(lam * q).mean()
                bc_loss = (alpha_actor * mse).mean()
                total_loss = actor_loss + bc_loss
                return total_loss, {
                    "loss": total_loss,
                    "actor_loss": actor_loss,
                    "bc_loss": bc_loss,
                    "mse": mse.mean(),
                    "q_mean": q.mean(),
                    "q_std": qs.std(),
                    "lambda": lam,
                }

            grads, log_dict = jax.grad(actor_loss_fn, has_aux=True)(actor_state.params)
            updates, actor_opt_state = actor_tx.update(grads, actor_state.opt_state, actor_state.params)
            actor_params = optax.apply_updates(actor_state.params, updates)
            return ActorRefitState(params=actor_params, opt_state=actor_opt_state, key=key), log_dict

        return actor_refit_step

    def make_initial_actor_state(self) -> ActorRefitState:
        return tree_to_device(
            ActorRefitState(
                params=deepcopy(self.initial_actor_params),
                opt_state=deepcopy(self.initial_actor_opt_state),
                key=deepcopy(self.initial_actor_key),
            ),
            self.device,
        )

    def fit_actor(
        self,
        replay_buffer: ReplayBuffer,
        actor_state: ActorRefitState,
        steps: int,
        batch_size: int,
        eval_env: Optional[Any] = None,
        eval_episodes: int = 0,
        eval_interval: int = 0,
        prefix: str = "actor_refit",
        save_dir: Optional[Union[str, Path]] = None,
        log_wandb: bool = False,
        log_every: int = 500,
        log_extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ActorRefitState, Dict[str, Any]]:
        refit_log: Dict[str, Any] = {
            f"{prefix}/final_loss": np.nan,
            f"{prefix}/final_bc_loss": np.nan,
            f"{prefix}/final_mse": np.nan,
            f"{prefix}/final_q_mean": np.nan,
            f"{prefix}/final_reward_mean": np.nan,
            f"{prefix}/final_success_rate": np.nan,
            f"{prefix}/best_reward_mean": np.nan,
            f"{prefix}/best_success_rate": np.nan,
            f"{prefix}/inner_eval_steps": [],
            f"{prefix}/inner_reward_mean": [],
            f"{prefix}/inner_success_rate": [],
        }
        if steps <= 0:
            return actor_state, refit_log

        best_eval_metric = -np.inf
        save_dir_path = Path(save_dir) if save_dir is not None else None
        if save_dir_path is not None:
            save_dir_path.mkdir(parents=True, exist_ok=True)
        log_extra = {} if log_extra is None else dict(log_extra)

        for fit_step in trange(1, int(steps) + 1, desc="Actor refit"):
            batch = replay_buffer.sample_batch(None, batch_size=batch_size)
            actor_state, step_log = self._actor_refit_step(actor_state, self.agent, batch)
            step_log = {key: float(jax.device_get(value)) for key, value in step_log.items()}
            refit_log[f"{prefix}/final_loss"] = step_log["loss"]
            refit_log[f"{prefix}/final_bc_loss"] = step_log["bc_loss"]
            refit_log[f"{prefix}/final_mse"] = step_log["mse"]
            refit_log[f"{prefix}/final_q_mean"] = step_log["q_mean"]

            if log_wandb and fit_step % max(1, int(log_every)) == 0:
                wandb.log({f"{prefix}/train/{key}": value for key, value in step_log.items()}, step=fit_step)

            should_eval = eval_env is not None and eval_episodes > 0 and eval_interval > 0 and (
                fit_step % eval_interval == 0 or fit_step == steps
            )
            if should_eval:
                eval_stats = self.eval_actor(eval_env, actor_state.params, n_episodes=eval_episodes)
                eval_log = eval_stats_to_eval_log(eval_stats, fit_step)
                reward_mean = eval_log["eval/reward_mean"]
                success_rate = eval_log["eval/success_rate"]
                refit_log[f"{prefix}/inner_eval_steps"].append(int(fit_step))
                refit_log[f"{prefix}/inner_reward_mean"].append(reward_mean)
                refit_log[f"{prefix}/inner_success_rate"].append(success_rate)
                refit_log[f"{prefix}/final_reward_mean"] = reward_mean
                refit_log[f"{prefix}/final_success_rate"] = success_rate
                metric = success_rate if np.isfinite(success_rate) else reward_mean
                is_best = np.isfinite(metric) and metric > best_eval_metric
                if is_best:
                    best_eval_metric = metric
                    refit_log[f"{prefix}/best_reward_mean"] = reward_mean
                    refit_log[f"{prefix}/best_success_rate"] = success_rate
                if save_dir_path is not None:
                    save_pickle(save_dir_path / "latest_actor.pkl", serialization.to_state_dict(actor_state.params))
                    if is_best:
                        save_pickle(save_dir_path / "best_actor.pkl", serialization.to_state_dict(actor_state.params))
                    save_logs_npz([{**log_extra, "refit_step": int(fit_step), **refit_log}], str(save_dir_path / "fit_eval_logs.npz"))
        return actor_state, refit_log

    def state_dict(self) -> Dict[str, Any]:
        return {
            "agent": serialization.to_state_dict(self.agent),
            "total_it": self.total_it,
            "initial_actor_params": serialization.to_state_dict(self.initial_actor_params),
            "initial_actor_opt_state": serialization.to_state_dict(self.initial_actor_opt_state),
            "initial_actor_key": serialization.to_state_dict(self.initial_actor_key),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        payload = state_dict["agent"] if "agent" in state_dict else state_dict
        self.agent = serialization.from_state_dict(self.agent, payload)
        self.total_it = int(state_dict.get("total_it", 0)) if isinstance(state_dict, dict) else 0
        if isinstance(state_dict, dict) and "initial_actor_params" in state_dict:
            self.initial_actor_params = serialization.from_state_dict(self.initial_actor_params, state_dict["initial_actor_params"])
        if isinstance(state_dict, dict) and "initial_actor_opt_state" in state_dict:
            self.initial_actor_opt_state = serialization.from_state_dict(self.initial_actor_opt_state, state_dict["initial_actor_opt_state"])
        if isinstance(state_dict, dict) and "initial_actor_key" in state_dict:
            self.initial_actor_key = serialization.from_state_dict(self.initial_actor_key, state_dict["initial_actor_key"])
        self.agent = tree_to_device(self.agent, self.device)
        self.initial_actor_params = tree_to_device(self.initial_actor_params, self.device)
        self.initial_actor_opt_state = tree_to_device(self.initial_actor_opt_state, self.device)
        self.initial_actor_key = tree_to_device(self.initial_actor_key, self.device)


def eval_stats_to_eval_log(eval_stats: Dict[str, Any], timestep: int) -> Dict[str, float]:
    reward_mean = float(eval_stats.get("episode.return", np.nan))
    normalized_score = float(eval_stats.get("episode.normalized_return", np.nan))
    success_rate = float(eval_stats.get("success", np.nan))
    return {
        "timestep": int(timestep),
        "eval/reward_mean": reward_mean,
        "eval/reward_std": np.nan,
        "eval/d4rl_normalized_score_mean": normalized_score,
        "eval/d4rl_normalized_score_std": np.nan,
        "eval/success_rate": success_rate,
        "eval/success_std": np.nan,
    }


def resolve_checkpoint_path(load_model: Union[str, Path], run_name: Optional[str] = None, seed: Optional[int] = None) -> Tuple[Path, Path]:
    load_path = Path(load_model)
    if load_path.is_file():
        if load_path.name not in ("checkpoint.pkl", "best_checkpoint.pkl"):
            raise FileNotFoundError(f"load_model points to a non-checkpoint file: {load_path}")
        return load_path.parent, load_path
    if not load_path.exists():
        raise FileNotFoundError(f"load_model path does not exist: {load_path}")
    for filename in ("checkpoint.pkl", "best_checkpoint.pkl"):
        direct = (load_path / filename).resolve()
        if direct.exists():
            return direct.parent, direct
    if run_name is not None and seed is not None:
        for filename in ("checkpoint.pkl", "best_checkpoint.pkl"):
            exact = (load_path / run_name / str(seed) / filename).resolve()
            if exact.exists():
                return exact.parent, exact
    candidates: List[Path] = []
    if run_name is not None:
        run_name_dir = load_path / run_name
        if run_name_dir.exists():
            candidates.extend(sorted(run_name_dir.glob("*/checkpoint.pkl")))
            candidates.extend(sorted(run_name_dir.glob("*/best_checkpoint.pkl")))
    candidates.extend(sorted(load_path.glob("*/checkpoint.pkl")))
    candidates.extend(sorted(load_path.glob("*/best_checkpoint.pkl")))
    candidates.extend(sorted(load_path.glob("*/*/checkpoint.pkl")))
    candidates.extend(sorted(load_path.glob("*/*/best_checkpoint.pkl")))
    existing = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            existing.append(candidate)
    if len(existing) == 0:
        raise FileNotFoundError(f"checkpoint file not found under: {load_path}")
    if len(existing) > 1:
        found = "\n".join(str(path) for path in existing)
        raise FileNotFoundError(f"Multiple checkpoint files found under {load_path}; provide a more specific path.\n{found}")
    return existing[0].parent, existing[0]


def load_run_config_for_actor_refit(current_config: TrainConfig, loaded_run_dir: Union[str, Path]) -> TrainConfig:
    loaded_run_dir = Path(loaded_run_dir)
    saved_config_path = loaded_run_dir / "config.yaml"
    if not saved_config_path.exists():
        raise FileNotFoundError(f"mode='actor_refit' expects saved run config at: {saved_config_path}")
    with open(saved_config_path, "r") as f:
        saved_raw = yaml.safe_load(f) or {}
    config_fields = set(TrainConfig.__dataclass_fields__.keys())
    saved_kwargs = {key: _coerce_hparam_value(value) for key, value in saved_raw.items() if key in config_fields}
    loaded_config = TrainConfig(**saved_kwargs)
    cli_overrides = _cli_overridden_fields()
    current_config_dict = asdict(current_config)
    for key in sorted(cli_overrides):
        if key in config_fields:
            setattr(loaded_config, key, current_config_dict[key])
    loaded_config.mode = "actor_refit"
    loaded_config.load_model = current_config.load_model
    loaded_config.checkpoints_path = None
    normalize_config_aliases(loaded_config)
    refresh_algorithm_names(loaded_config)
    validate_config(loaded_config)
    print(f"Loaded saved run config for actor_refit from: {saved_config_path}")
    return loaded_config


def make_checkpoint_payload(trainer: DDReBRACFQLJAX, config: TrainConfig, state_mean: Union[np.ndarray, float], state_std: Union[np.ndarray, float]) -> Dict[str, Any]:
    return {
        "trainer": trainer.state_dict(),
        "config": asdict(config),
        "state_mean": state_mean,
        "state_std": state_std,
    }


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    trainer: DDReBRACFQLJAX,
    config: TrainConfig,
    state_mean: Union[np.ndarray, float],
    state_std: Union[np.ndarray, float],
    log_wandb: bool,
) -> None:
    save_pickle(checkpoint_path, make_checkpoint_payload(trainer=trainer, config=config, state_mean=state_mean, state_std=state_std))
    # if log_wandb and wandb.run is not None:
    #     wandb.save(str(checkpoint_path), policy="now")


def _train_impl(config: TrainConfig):
    normalize_config_aliases(config)
    actor_refit_only = config.mode == "actor_refit"
    loaded_run_dir: Optional[Path] = None
    checkpoint_path: Optional[Path] = None

    if actor_refit_only:
        loaded_run_dir, checkpoint_path = resolve_checkpoint_path(config.load_model, run_name=config.name, seed=config.seed)
        config = load_run_config_for_actor_refit(config, loaded_run_dir)
    else:
        config = apply_env_hyperparams(config)
        config = finalize_checkpoint_path(config)

    checkpoint_manager = None
    checkpoint_preparation = None
    if config.checkpoints_path is not None and (not actor_refit_only):
        current_config_dict = asdict(config)
        checkpoint_manager = TrainingCheckpointManager(
            run_dir=config.checkpoints_path,
            current_config=current_config_dict,
            default_config=asdict(TrainConfig()),
            max_timesteps=int(config.max_timesteps),
            checkpoint_type="dd_rebrac_jax_training_progress",
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

    replay_buffer = ReplayBuffer(device=jax_device)
    replay_buffer.create_from_dataset(
        env_name=config.env,
        dataset=train_dataset,
        normalize_reward=config.normalize_reward,
        is_normalize=config.normalize_states,
        p_aug=config.p_aug,
        frame_stack=config.frame_stack,
    )
    if val_dataset is not None:
        val_buffer = ReplayBuffer(device=jax_device)
        val_buffer.create_from_dataset(
            env_name=config.env,
            dataset=val_dataset,
            normalize_reward=config.normalize_reward,
            is_normalize=config.normalize_states,
            p_aug=config.p_aug,
            frame_stack=config.frame_stack,
        )
    else:
        val_buffer = None

    state_mean, state_std = replay_buffer.mean, replay_buffer.std
    if config.normalize_states:
        eval_env = TransformEnv(eval_env, state_mean=state_mean, state_std=state_std)
        print("Warning: normalize_states=True changes the original FQL/OGBench ReBRAC codepath.")
    set_seed(config.seed, eval_env)


    print("---------------------------------------")
    run_mode_name = "Actor refit" if actor_refit_only else "Training"
    print(f"{run_mode_name} {ALGORITHM_NAME}-JAX-FQL, Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")

    example_batch = replay_buffer.dataset.sample(1)
    trainer = DDReBRACFQLJAX(
        config=config,
        ex_observations=example_batch["observations"],
        ex_actions=example_batch["actions"],
        device=jax_device,
    )

    if config.load_model != "":
        if checkpoint_path is None or loaded_run_dir is None:
            loaded_run_dir, checkpoint_path = resolve_checkpoint_path(config.load_model, run_name=config.name, seed=config.seed)
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = load_pickle(checkpoint_path)
        if isinstance(checkpoint, dict) and "trainer" in checkpoint:
            trainer.load_state_dict(checkpoint["trainer"])
            state_mean = checkpoint.get("state_mean", state_mean)
            state_std = checkpoint.get("state_std", state_std)
            if config.normalize_states:
                eval_env = TransformEnv(eval_env.env if isinstance(eval_env, TransformEnv) else eval_env, state_mean=state_mean, state_std=state_std)
        else:
            trainer.load_state_dict(checkpoint)

    def _progress_state():
        return make_checkpoint_payload(
            trainer=trainer,
            config=config,
            state_mean=state_mean,
            state_std=state_std,
        )

    def _final_state():
        return _progress_state()

    def _load_progress_state(payload):
        raw_trainer = payload["trainer"] if isinstance(payload, dict) and "trainer" in payload else payload
        trainer.load_state_dict(raw_trainer)

    def _training_timestep():
        return int(trainer.total_it)

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


    if actor_refit_only:
        actor_refit_dir = loaded_run_dir / config.actor_refit_dir_name
        actor_refit_dir.mkdir(parents=True, exist_ok=True)
        actor_state = trainer.make_initial_actor_state()
        refit_actor_state, refit_log = trainer.fit_actor(
            replay_buffer=replay_buffer,
            actor_state=actor_state,
            steps=int(config.max_timesteps),
            batch_size=config.batch_size,
            eval_env=eval_env,
            eval_episodes=config.n_episodes,
            eval_interval=config.eval_freq,
            prefix="actor_refit",
            save_dir=actor_refit_dir,
            log_wandb=config.log_wandb,
            log_every=config.log_every,
            log_extra={"loaded_checkpoint": str(checkpoint_path)},
        )
        save_pickle(actor_refit_dir / "final_actor.pkl", serialization.to_state_dict(refit_actor_state.params))
        save_logs_npz([{"loaded_checkpoint": str(checkpoint_path), **refit_log}], str(actor_refit_dir / "fit_eval_logs.npz"))
        with open(actor_refit_dir / "refit_config.yaml", "w") as f:
            pyrallis.dump(config, f)
        print("Actor refit finished")
        return

    best_eval_metric_mean = best_eval_metric(eval_logs)
    last_train_log: Dict[str, float] = {}

    def _evaluation_required(step):
        step = int(step)
        return (
            (config.eval_at_first_step and step == 1)
            or step % int(config.eval_freq) == 0
            or step == int(config.max_timesteps)
        )

    try:
        for t in trange(start_timestep, int(config.max_timesteps), desc=f"{ALGORITHM_NAME}-FQL Training"):
            i = t + 1
            batch = replay_buffer.sample_batch(None, batch_size=config.batch_size)
            update_actor_now = (i % config.policy_freq) == 0
            log_dict = trainer.train(batch, update_actor_now=update_actor_now)
            train_step = int(trainer.total_it)
            last_train_log = {**last_train_log, **log_dict}
    
            if config.log_wandb and train_step % config.log_every == 0:
                _wandb_log({f"train/{key}": value for key, value in last_train_log.items()}, train_step)
    
            if val_buffer is not None and train_step % config.log_every == 0:
                val_batch = val_buffer.sample_batch(None, batch_size=config.batch_size)
                _, val_info = trainer.agent.total_loss(val_batch, grad_params=None, full_update=True)
                val_info = {f"validation/{key}": float(jax.device_get(value)) for key, value in val_info.items()}
                if config.log_wandb:
                    _wandb_log(val_info, train_step)
    
            should_eval = (
                (config.eval_at_first_step and train_step == 1)
                or train_step % config.eval_freq == 0
                or train_step == int(config.max_timesteps)
            )
            if should_eval:
                print(f"Time steps: {train_step}")
                eval_stats = trainer.eval_current_actor(eval_env, n_episodes=config.n_episodes)
                eval_log = eval_stats_to_eval_log(eval_stats, train_step)
                upsert_eval_log(eval_logs, eval_log)
                print(
                    f"Evaluation over {config.n_episodes} episodes: "
                    f"reward={eval_log['eval/reward_mean']:.3f}, "
                    f"d4rl_normalized={eval_log['eval/d4rl_normalized_score_mean']:.3f}, "
                    f"success_rate={eval_log['eval/success_rate']:.3f}"
                )
                if config.log_wandb:
                    _wandb_log({key: to_python_scalar(value) for key, value in eval_log.items() if is_scalar_value(value)}, train_step)
                save_and_upload_eval_logs(eval_logs, config.checkpoints_path, config.log_wandb)
                if config.checkpoints_path is not None and config.save_best_model:
                    metric = eval_log["eval/success_rate"] if np.isfinite(eval_log["eval/success_rate"]) else eval_log["eval/d4rl_normalized_score_mean"]
                    if not np.isfinite(metric):
                        metric = eval_log["eval/reward_mean"]
                    is_best = np.isfinite(metric) and metric > best_eval_metric_mean
                    if is_best:
                        best_eval_metric_mean = metric
                        save_checkpoint(
                            os.path.join(config.checkpoints_path, "best_checkpoint.pkl"),
                            trainer=trainer,
                            config=config,
                            state_mean=state_mean,
                            state_std=state_std,
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
