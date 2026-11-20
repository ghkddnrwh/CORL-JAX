"""Official DriftQL algorithm + ORL-SMOOTH experiment interface.

The mathematical algorithm, network, optimizer, dataset sampling, and evaluation
code live in ``driftql_original`` and are vendored from driftql-main.zip. The
only edits within those modules are fully qualified import paths.

This runner deliberately implements the *offline* setting. Online fine-tuning
requires preservation of the mutable simulator and online replay-buffer state
for faithful interrupted-run recovery and is not silently approximated here.
"""

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')

import pickle
import random
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
    wandb = None

_PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / 'algorithms').is_dir()
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.uk_offline.common.checkpointing import (
    DEFAULT_IDENTITY_IGNORED_FIELDS,
    TrainingCheckpointManager,
    best_eval_metric,
    find_eval_log,
    log_training_exceptions,
    upsert_eval_log,
)
from algorithms.uk_offline.driftql_original.agents.driftql import DriftQLAgent
from algorithms.uk_offline.driftql_original.envs.env_utils import make_env_and_datasets
from algorithms.uk_offline.driftql_original.utils.datasets import Dataset, ReplayBuffer
from algorithms.uk_offline.driftql_original.utils.evaluation import evaluate as original_evaluate

ALGORITHM_NAME = 'DriftQL'
ALGORITHM_FULL_NAME = 'Drift Q-Learning, official update, ORL-SMOOTH experiment interface'


@dataclass
class TrainConfig:
    # ORL experiment parameters.
    device: str = 'gpu'
    env: str = 'cube-double-play-singletask-v0'
    dataset_name: Optional[str] = None
    seed: int = 0
    eval_seed: int = 42
    max_timesteps: int = 1_000_000
    eval_freq: int = 100_000
    n_episodes: int = 50
    checkpoints_path: Optional[str] = None
    load_model: str = ''
    mode: str = 'train'
    hyperparams_path: Optional[str] = 'hyperparams/driftql_jax.yml'
    use_hyperparams: bool = True
    run_tag: str = ''

    # Original DriftQL dataset and schedule behavior.
    batch_size: int = 256
    buffer_size: int = 2_000_000
    offline_dataset_path: Optional[str] = None
    frame_stack: Optional[int] = None
    p_aug: Optional[float] = None
    action_clip_eps: Optional[float] = 1e-5
    normalize_states: bool = False
    normalize_reward: bool = False
    online_steps: int = 0

    # Original DriftQL optimizer and network (one shared Adam optimizer).
    lr: Optional[float] = None
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4  # Compatibility field; original uses one lr.
    hidden_dim: int = 512
    actor_n_hiddens: int = 4
    critic_n_hiddens: int = 4
    actor_hidden_dims: Optional[Tuple[int, ...]] = None
    value_hidden_dims: Optional[Tuple[int, ...]] = None
    layer_norm: Optional[bool] = None
    actor_layer_norm: Optional[bool] = None
    critic_ln: bool = True
    actor_ln: bool = False
    encoder: Optional[str] = None

    # Original DriftQL mathematical hyperparameters.
    discount: float = 0.99
    tau: float = 0.005
    q_agg: str = 'min'  # Bellman target aggregation.
    actor_critic_aggregation: str = 'mean'  # original q_agg_actor
    alpha: float = 10.0
    normalize_q_loss: bool = False
    kernel: str = 'laplace'
    drift_temp: float = 0.2
    dim_scale: bool = True
    drift_ngen: int = 32
    noise_dim: int = 0  # Original: <=0 resolves to action dimension.
    drift_eta: float = 1.0
    drift_normalize: bool = False
    drift_batch_size: int = 256
    drift_eps: float = 1e-12

    # ORL W&B / checkpoint interface.
    project: str = 'ORL-SMOOTH'
    group: str = 'DriftQL-JAX'
    name: str = 'DriftQL-JAX'
    log_wandb: bool = True
    log_every: int = 5000
    checkpoint_freq: int = 25_000
    save_best_model: bool = True
    save_final_model: bool = True
    eval_at_first_step: bool = True
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        if self.dataset_name is not None:
            self.env = self.dataset_name
        refresh_algorithm_names(self)
        validate_config(self)


def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = 'ORL-SMOOTH'
    config.group = f'{ALGORITHM_NAME}-JAX'
    if (config.q_agg, config.actor_critic_aggregation) == ('min', 'min'):
        agg_suffix = ''
    elif (config.q_agg, config.actor_critic_aggregation) == ('mean', 'mean'):
        agg_suffix = '-QMEAN'
    else:
        agg_suffix = f'-T{config.q_agg.upper()}-A{config.actor_critic_aggregation.upper()}'
    if config.run_tag:
        if not all(c.isalnum() or c in ('-', '_') for c in config.run_tag):
            raise ValueError('run_tag only allows letters, digits, underscore, hyphen')
        agg_suffix += '-' + config.run_tag
    config.name = f'{ALGORITHM_NAME}-JAX-{config.env}{agg_suffix}'


def validate_config(config: TrainConfig) -> None:
    if config.mode != 'train':
        raise NotImplementedError("The original DriftQL actor is updated jointly. Only --mode train is supported.")
    if config.online_steps:
        raise NotImplementedError('This ORL adapter supports original offline DriftQL only (online_steps=0).')
    if config.normalize_states or config.normalize_reward:
        raise ValueError('Original DriftQL offline protocol does not apply state/reward normalization; use False.')
    if config.env.strip() == '':
        raise ValueError('env must be nonempty')
    if config.max_timesteps < 0 or min(config.eval_freq, config.n_episodes, config.log_every, config.checkpoint_freq, config.batch_size, config.buffer_size) <= 0:
        raise ValueError('Training/evaluation/checkpoint intervals and batch/buffer sizes must be positive.')
    if config.q_agg not in ('mean', 'min') or config.actor_critic_aggregation not in ('mean', 'min'):
        raise ValueError('q_agg and actor_critic_aggregation must be min or mean')
    if config.kernel not in ('laplace', 'gaussian'):
        raise ValueError('kernel must be laplace or gaussian')
    if not (0 <= config.discount <= 1 and 0 <= config.tau <= 1):
        raise ValueError('discount and tau must be in [0,1]')
    if config.alpha < 0 or config.drift_temp <= 0 or config.drift_ngen < 1 or config.drift_batch_size < 1 or config.drift_eps <= 0:
        raise ValueError('Invalid DriftQL coefficient / temperature / sample count')
    if config.drift_eta < 0:
        raise ValueError('drift_eta must be >=0')
    if config.p_aug is not None and not (0 <= config.p_aug <= 1):
        raise ValueError('p_aug must be in [0,1]')
    if config.frame_stack is not None and config.frame_stack < 1:
        raise ValueError('frame_stack must be positive')
    if config.lr is not None and config.lr <= 0:
        raise ValueError('lr must be >0')
    if config.actor_learning_rate <= 0 or config.critic_learning_rate <= 0:
        raise ValueError('learning rates must be >0')


def _cli_overridden_fields(args=None):
    args = sys.argv[1:] if args is None else args
    out = set()
    for token in args:
        if token.startswith('--'):
            out.add(token[2:].split('=', 1)[0].replace('-', '_'))
    return out


def _coerce(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return tuple(value)
    return value


def apply_env_hyperparams(config: TrainConfig) -> TrainConfig:
    """Precedence: TrainConfig < environment YAML < explicitly supplied CLI."""
    overrides = _cli_overridden_fields()
    yaml_settings = {}
    if config.use_hyperparams and config.hyperparams_path is not None:
        yaml_path = Path(config.hyperparams_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f'Hyperparameter YAML missing: {yaml_path}')
        with yaml_path.open(encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        if config.env not in data:
            print(f"Warning: no official YAML hyperparameters for '{config.env}'; using TrainConfig/CLI defaults.")
        else:
            yaml_settings = data[config.env] or {}
            aliases = {'q_agg_actor': 'actor_critic_aggregation', 'offline_steps': 'max_timesteps',
                       'eval_interval': 'eval_freq', 'eval_episodes': 'n_episodes',
                       'log_interval': 'log_every', 'env_name': 'env'}
            known = TrainConfig.__dataclass_fields__
            for raw_key, value in yaml_settings.items():
                key = aliases.get(raw_key, raw_key)
                if key not in known:
                    print(f'Ignored unknown DriftQL YAML key: {raw_key}')
                    continue
                if key in overrides or raw_key in overrides:
                    continue
                # If CLI explicitly overrides the common rate, don't let YAML's
                # equivalent original rate shadow the CLI value.
                if key == 'lr' and 'actor_learning_rate' in overrides:
                    continue
                if key in ('actor_hidden_dims', 'value_hidden_dims') and 'hidden_dim' in overrides:
                    continue
                if key == 'actor_hidden_dims' and 'actor_n_hiddens' in overrides:
                    continue
                if key == 'value_hidden_dims' and 'critic_n_hiddens' in overrides:
                    continue
                if key == 'layer_norm' and 'critic_ln' in overrides:
                    continue
                if key == 'actor_layer_norm' and 'actor_ln' in overrides:
                    continue
                setattr(config, key, _coerce(value))
    # The original uses a *single* Adam optimizer, while other ORL modules
    # expose distinct actor/critic names. Resolve aliases without changing math.
    if 'lr' in overrides:
        config.actor_learning_rate = float(config.lr)
    elif 'actor_learning_rate' in overrides:
        config.lr = float(config.actor_learning_rate)
    elif yaml_settings.get('lr') is not None and yaml_settings.get('actor_learning_rate') is None:
        config.actor_learning_rate = float(config.lr)
    else:
        config.lr = float(config.actor_learning_rate)
    if config.actor_hidden_dims is None:
        config.actor_hidden_dims = tuple([config.hidden_dim] * config.actor_n_hiddens)
    else:
        config.actor_hidden_dims = tuple(config.actor_hidden_dims)
    if config.value_hidden_dims is None:
        config.value_hidden_dims = tuple([config.hidden_dim] * config.critic_n_hiddens)
    else:
        config.value_hidden_dims = tuple(config.value_hidden_dims)
    if config.layer_norm is None or 'critic_ln' in overrides:
        config.layer_norm = bool(config.critic_ln)
    if config.actor_layer_norm is None or 'actor_ln' in overrides:
        config.actor_layer_norm = bool(config.actor_ln)
    if config.critic_learning_rate != config.lr:
        print(f'Warning: original DriftQL has one shared Adam optimizer; critic_learning_rate={config.critic_learning_rate} is not used; using lr={config.lr}.')
    refresh_algorithm_names(config)
    validate_config(config)
    return config


def config_to_agent_config(config: TrainConfig) -> Dict[str, Any]:
    """Map ORL names into exact upstream get_config keys; no algorithm changes."""
    return dict(
        agent_name='driftql',
        lr=float(config.lr),
        batch_size=int(config.batch_size),
        actor_hidden_dims=tuple(config.actor_hidden_dims),
        value_hidden_dims=tuple(config.value_hidden_dims),
        noise_dim=int(config.noise_dim),
        layer_norm=bool(config.layer_norm),
        actor_layer_norm=bool(config.actor_layer_norm),
        discount=float(config.discount), tau=float(config.tau),
        q_agg=config.q_agg, q_agg_actor=config.actor_critic_aggregation,
        alpha=float(config.alpha), normalize_q_loss=bool(config.normalize_q_loss),
        kernel=config.kernel, drift_temp=float(config.drift_temp),
        dim_scale=bool(config.dim_scale), drift_ngen=int(config.drift_ngen),
        drift_eta=float(config.drift_eta), drift_normalize=bool(config.drift_normalize),
        drift_batch_size=int(config.drift_batch_size), drift_eps=float(config.drift_eps),
        encoder=config.encoder,
    )


def finalize_checkpoint_path(config):
    if config.checkpoints_path is not None:
        config.checkpoints_path = str(Path(config.checkpoints_path) / config.name / str(config.seed))
    return config


def select_jax_device(device):
    backend = 'gpu' if device.lower() == 'cuda' else device.lower()
    try:
        dev = jax.devices(backend)[0]
    except Exception:
        print(f"Requested JAX backend {backend!r} unavailable; falling back to default.")
        dev = jax.devices()[0]
    print(f'Using JAX device: {dev}')
    return dev


def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if env is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, 'seed'):
                env.seed(seed)
        if hasattr(env.action_space, 'seed'):
            env.action_space.seed(seed)


def save_pickle(path, payload):
    with Path(path).open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)


def resolve_checkpoint_path(load_model, run_name=None, seed=None):
    root = Path(load_model)
    if root.is_file():
        return root
    if not root.exists():
        raise FileNotFoundError(f'--load_model path does not exist: {root}')
    names = ('checkpoint.pkl', 'best_checkpoint.pkl')
    for name in names:
        f = root / name
        if f.is_file():
            return f
    if run_name is not None and seed is not None:
        for name in names:
            f = root / run_name / str(seed) / name
            if f.is_file():
                return f
    candidates = sorted({p.resolve() for pat in ('*/checkpoint.pkl','*/best_checkpoint.pkl','*/*/checkpoint.pkl','*/*/best_checkpoint.pkl') for p in root.glob(pat)})
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one checkpoint under {root}, found {len(candidates)}. Pass an exact --load_model file.')
    return candidates[0]


def _finite_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return (float(np.mean(values)), float(np.std(values))) if values.size else (float('nan'), float('nan'))


def _save_npz(logs: List[Dict[str, Any]], path):
    if not logs or path is None:
        return
    keys = list(dict.fromkeys(key for log in logs for key in log))
    arrays = {key: np.asarray([log.get(key, np.nan) for log in logs]) for key in keys}
    np.savez(path, **arrays)


def evaluate_agent(agent, eval_env, n_episodes, eval_seed, timestep):
    """Call the upstream evaluator unchanged; only aggregate its episode records.

    Evaluation RNG is isolated from the training RNG (a runner-level change).
    No action ranking, greedy selection, or sampling alteration is introduced.
    """
    numpy_state = np.random.get_state()
    try:
        np.random.seed((int(eval_seed) + int(timestep)) % (2 ** 32 - 1))
        metrics, trajs, _ = original_evaluate(
            agent=agent, env=eval_env, config=agent.config,
            num_eval_episodes=int(n_episodes), num_video_episodes=0,
            eval_temperature=0,
        )
    finally:
        np.random.set_state(numpy_state)
    returns = [sum(float(r) for r in t['reward']) for t in trajs]
    success = []
    normalized = []
    for traj in trajs:
        info = traj['info'][-1]
        ep_info = info.get('episode', {})
        success.append(info.get('success', ep_info.get('success', np.nan)))
        normalized.append(ep_info.get('normalized_return', np.nan))
    reward_mean, reward_std = _finite_mean_std(returns)
    score_mean, score_std = _finite_mean_std(normalized)
    success_mean, success_std = _finite_mean_std(success)
    if not np.isfinite(success_mean) and 'success' in metrics:
        success_mean = float(metrics['success'])
    if not np.isfinite(score_mean) and 'episode.normalized_return' in metrics:
        score_mean = float(metrics['episode.normalized_return'])
    return dict(timestep=int(timestep), **{
        'eval/reward_mean': reward_mean,
        'eval/reward_std': reward_std,
        'eval/d4rl_normalized_score_mean': score_mean,
        'eval/d4rl_normalized_score_std': score_std,
        'eval/success_rate': success_mean,
        'eval/success_std': success_std,
    })


def _train_impl(config: TrainConfig):
    config = apply_env_hyperparams(config)
    config = finalize_checkpoint_path(config)
    checkpoint_manager = None
    preparation = None
    if config.checkpoints_path is not None:
        checkpoint_manager = TrainingCheckpointManager(
            run_dir=config.checkpoints_path, current_config=asdict(config),
            default_config=asdict(TrainConfig()), max_timesteps=int(config.max_timesteps),
            checkpoint_type='driftql_jax_training_progress',
            identity_ignored_fields=DEFAULT_IDENTITY_IGNORED_FIELDS,
            checkpoint_version=2, accepted_checkpoint_versions=(1, 2),
            wandb_enabled=bool(config.log_wandb), wandb_entity=config.wandb_entity,
            wandb_project=config.project, final_checkpoint_name='checkpoint.pkl',
        )
        preparation = checkpoint_manager.prepare()
        print(preparation.message)
        if preparation.is_completed:
            return

    device = select_jax_device(config.device)
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        config.env, frame_stack=config.frame_stack,
        action_clip_eps=config.action_clip_eps,
        offline_dataset_path=config.offline_dataset_path,
    )
    set_seed(config.seed, eval_env)

    # Exactly the original main.py offline replay-buffer construction.
    train_dataset = Dataset.create(**train_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(config.buffer_size, train_dataset.size + 1)
    )
    replay_buffer = train_dataset
    for dataset in (train_dataset, val_dataset, replay_buffer):
        if dataset is not None:
            dataset.p_aug = config.p_aug
            dataset.frame_stack = config.frame_stack
            dataset.return_next_actions = False
    example_batch = train_dataset.sample(1)
    agent = DriftQLAgent.create(
        config.seed, example_batch['observations'], example_batch['actions'],
        config_to_agent_config(config),
    )
    agent = jax.device_put(agent, device)
    train_timestep = 0
    train_logs: List[Dict[str, Any]] = []

    def current_state():
        return {'agent': serialization.to_state_dict(agent),
                'config': asdict(config), '_training_timestep': int(train_timestep),
                'numpy_rng_state': np.random.get_state(), 'python_rng_state': random.getstate(),
                'train_logs': train_logs}

    def load_state(payload):
        nonlocal agent, train_timestep, train_logs
        if not isinstance(payload, dict) or 'agent' not in payload:
            raise ValueError('Checkpoint must contain the official DriftQL agent state.')
        agent = serialization.from_state_dict(agent, payload['agent'])
        agent = jax.device_put(agent, device)
        train_timestep = int(payload.get('_training_timestep', 0))
        train_logs = list(payload.get('train_logs', []))
        if 'numpy_rng_state' in payload:
            np.random.set_state(payload['numpy_rng_state'])
        if 'python_rng_state' in payload:
            random.setstate(payload['python_rng_state'])

    if config.load_model:
        ckpt_file = resolve_checkpoint_path(config.load_model, config.name, config.seed)
        print(f'Loading model state from: {ckpt_file}')
        load_state(load_pickle(ckpt_file))

    eval_logs: List[Dict[str, Any]] = []
    if checkpoint_manager is not None:
        if preparation.is_resuming:
            start, eval_logs, _ = checkpoint_manager.restore(
                load_trainer_state=load_state, get_restored_timestep=lambda: train_timestep,
            )
            print(f'Restored training at step {start}.')
        else:
            checkpoint_manager.save_progress(
                timestep=train_timestep, trainer_state=current_state(),
                eval_logs=eval_logs, status='running',
            )

    if config.log_wandb:
        if wandb is None:
            print('Warning: wandb not installed; continuing with local logs only.')
            config.log_wandb = False
        elif checkpoint_manager is not None:
            if not checkpoint_manager.initialize_wandb_with_fallback(
                    wandb_module=wandb, config=asdict(config), code_root=_PROJECT_ROOT):
                config.log_wandb = False
        else:
            try:
                run = wandb.init(project=config.project, group=config.group,
                                 name=config.name, entity=config.wandb_entity,
                                 config=asdict(config), id=str(uuid.uuid4()))
                if run is not None and hasattr(run, 'log_code'):
                    run.log_code(str(_PROJECT_ROOT))
            except Exception as exc:
                print(f'Warning: W&B unavailable: {exc}; continuing locally.')
                config.log_wandb = False

    def log_wandb(metrics, step):
        if not config.log_wandb:
            return
        if checkpoint_manager is not None:
            checkpoint_manager.log_wandb(metrics, int(step))
        else:
            wandb.log(metrics, step=int(step))

    def eval_due(step):
        return ((config.eval_at_first_step and step == 1) or
                step % config.eval_freq == 0 or step == config.max_timesteps)

    best_score = best_eval_metric(eval_logs)
    first_time = last_time = time.time()
    try:
        for i in trange(train_timestep + 1, int(config.max_timesteps) + 1, desc='DriftQL Training'):
            batch = train_dataset.sample(config.batch_size)
            # ORIGINAL, UNMODIFIED mathematical gradient / optimizer update:
            agent, update_info = agent.update(batch)
            train_timestep = int(i)

            if i % config.log_every == 0:
                metrics = {f'training/{k}': float(jax.device_get(v))
                           for k, v in update_info.items()}
                if val_dataset is not None:
                    val_batch = val_dataset.sample(config.batch_size)
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                    metrics.update({f'validation/{k}': float(jax.device_get(v))
                                    for k, v in val_info.items()})
                metrics['time/epoch_time'] = (time.time() - last_time) / config.log_every
                metrics['time/total_time'] = time.time() - first_time
                last_time = time.time()
                log_wandb(metrics, i)
                train_logs.append({'timestep': i, **metrics})
                if config.checkpoints_path is not None:
                    _save_npz(train_logs, Path(config.checkpoints_path) / 'train_logs.npz')

            if eval_due(i):
                result = evaluate_agent(agent, eval_env, config.n_episodes, config.eval_seed, i)
                upsert_eval_log(eval_logs, result)
                print(f"Evaluation at step {i}: reward={result['eval/reward_mean']:.3f} "
                      f"± {result['eval/reward_std']:.3f}; "
                      f"success={result['eval/success_rate']:.3f} "
                      f"± {result['eval/success_std']:.3f}")
                log_wandb(result, i)
                if config.checkpoints_path is not None:
                    _save_npz(eval_logs, Path(config.checkpoints_path) / 'eval_logs.npz')
                if config.checkpoints_path and config.save_best_model:
                    score = result['eval/success_rate']
                    if not np.isfinite(score): score = result['eval/d4rl_normalized_score_mean']
                    if not np.isfinite(score): score = result['eval/reward_mean']
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        save_pickle(Path(config.checkpoints_path) / 'best_checkpoint.pkl', current_state())

            if checkpoint_manager is not None and i % config.checkpoint_freq == 0:
                checkpoint_manager.save_progress(
                    timestep=i, trainer_state=current_state(), eval_logs=eval_logs,
                    status='running',
                )
    except BaseException:
        if checkpoint_manager is not None:
            interrupted_step = int(train_timestep)
            eval_committed = (not eval_due(interrupted_step) or
                              find_eval_log(eval_logs, interrupted_step) is not None)
            if eval_committed:
                checkpoint_manager.save_progress(
                    timestep=interrupted_step, trainer_state=current_state(),
                    eval_logs=eval_logs, status='interrupted',
                )
                print(f'Saved interrupted checkpoint at step {interrupted_step}.')
            else:
                print('Evaluation interrupted; retaining earlier safe progress checkpoint.')
        raise

    if checkpoint_manager is not None:
        path = checkpoint_manager.complete(
            timestep=int(train_timestep), final_state=current_state(),
            save_final_model=bool(config.save_final_model), eval_logs=eval_logs,
        )
        if path is not None:
            print(f'Saved final model: {path}')


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
        if wandb is not None and getattr(wandb, 'run', None) is not None:
            wandb.finish(exit_code=exit_code)


if __name__ == '__main__':
    train()
