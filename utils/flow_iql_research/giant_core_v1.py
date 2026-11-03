"""Antmaze-giant adapters; reuse frozen algorithms without changing equations."""
from pathlib import Path
from utils.flow_iql_research import core as iql_core
from utils.flow_iql_research import benchmark_core as fql_core
from utils.flow_iql_research import mean_empirical_backup_core_v1 as proposed_core

ENVIRONMENT = 'antmaze-giant-navigate-singletask-v0'
KINDS = ('iql', 'fql', 'mean_backup')
V_UPDATES = {'iql': 1, 'fql': 0, 'mean_backup': 2}


def config_for(environment, kind, discount, expectile=.9):
    assert environment == ENVIRONMENT and kind in KINDS and discount in (.99, .999)
    if kind == 'iql':
        return iql_core.config_for(environment, discount, expectile)
    if kind == 'fql':
        return fql_core.config_for(environment, discount, kind='fql')
    cfg = proposed_core.config_for(environment, discount, expectile)
    cfg.update(num_samples=9, flow_steps=10, agent_name='ifql', policy_type='ifql')
    return cfg


def bounds(environment, discount, reward_min, reward_max):
    assert environment == ENVIRONMENT and -1 <= reward_min <= reward_max <= 0
    # Installed OGBench relabel_dataset uses success-1 for all ant mazes.
    return -1/(1-discount), 0.


def module(kind):
    return {'iql': iql_core, 'fql': fql_core, 'mean_backup': proposed_core}[kind]


def agent_kind(kind):
    return 'ifql' if kind == 'mean_backup' else kind


def create(seed, observations, actions, cfg, kind):
    return module(kind).create(seed, observations, actions, cfg, agent_kind(kind))


def build_update(kind, interval):
    profile = 'bounded_v2_mean_empirical_backup' if kind == 'mean_backup' else 'official'
    return module(kind).build_update(profile, interval, agent_kind(kind))


make_scan = iql_core.make_scan


def probe(state, batch, kind):
    return module(kind).probe(state, batch)


def summarize(values, cfg, interval, kind):
    alpha = cfg.get('alpha') if kind == 'iql' else None
    return module(kind).probe_summary(values, cfg.get('expectile'), interval, alpha)


def source_paths():
    folder = Path(__file__).resolve().parent
    paths = [Path(__file__).resolve()]
    paths += [folder/name for name in ('core.py', 'benchmark_core.py', 'policy_core.py',
        'mean_empirical_backup_core_v1.py', 'selection_v2.py', 'train_mean_empirical_backup_v1.py')]
    paths += sorted((folder/'vendor').glob('*.py'))
    paths += [folder.parent/'propagation_research'/name for name in ('seeded_reset.py', 'reward_bounds.py')]
    return paths
