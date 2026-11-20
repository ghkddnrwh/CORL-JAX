"""Forty non-default OGBench tasks, with the frozen N9 learning equations.

The underlying trajectories are shared within a family. OGBench relabels
rewards and bootstrap masks for the explicitly requested single task.
"""
from pathlib import Path

from utils.codex.ukjo1.error_propagation.flow_iql_research import sample_budget_core_v1 as base


FAMILIES = (
    ('aml', 'antmaze-large-navigate-singletask-v0', 1),
    ('amg', 'antmaze-giant-navigate-singletask-v0', 1),
    ('hmm', 'humanoidmaze-medium-navigate-singletask-v0', 1),
    ('hml', 'humanoidmaze-large-navigate-singletask-v0', 1),
    ('soccer', 'antsoccer-arena-navigate-singletask-v0', 4),
    ('cube1', 'cube-single-play-singletask-v0', 2),
    ('cube2', 'cube-double-play-singletask-v0', 2),
    ('puzzle3', 'puzzle-3x3-play-singletask-v0', 4),
    ('puzzle4', 'puzzle-4x4-play-singletask-v0', 4),
    ('scene', 'scene-play-singletask-v0', 2),
)
TASKS = tuple(dict(code=code, family=family, default_task=default, task_id=task,
                   environment=family.replace('-singletask-v0', f'-singletask-task{task}-v0'))
              for code, family, default in FAMILIES for task in range(1, 6) if task != default)
ENVIRONMENTS = tuple(t['environment'] for t in TASKS)
RECIPES = ('g995_n9', 'g999_n9')
DISCOUNTS = {'g995_n9': .995, 'g999_n9': .999}
create = base.create
make_scan = base.make_scan
actor_metadata = base.actor_metadata
export_actor = base.export_actor
make_actor_policy = base.make_actor_policy
make_teacher_policy = base.make_teacher_policy
probe_summary = base.probe_summary
lean = base.lean


def task_info(environment):
    return next(t for t in TASKS if t['environment'] == environment)


def family_of(environment):
    if environment in base.ENVIRONMENTS:
        return environment
    return task_info(environment)['family']


def dataset_name(environment):
    return family_of(environment).replace('-singletask', '')


def config_for(environment, recipe, width=None):
    assert recipe in RECIPES
    family = family_of(environment)
    cfg = base.config_for(family, 'n9', width)
    cfg['discount'] = DISCOUNTS[recipe]
    cfg['onestep_specification'].update(name=recipe, discount=cfg['discount'])
    return cfg


def return_bounds(environment, discount, reward_min, reward_max, source='ogbench'):
    return base.return_bounds(family_of(environment), discount, reward_min, reward_max, source)


def build_update(recipe, bounds):
    assert recipe in RECIPES
    return base.build_update('n9', bounds)


def make_probe(recipe, bounds):
    assert recipe in RECIPES
    return base.make_probe('n9', bounds)


def source_paths():
    return [Path(__file__).resolve()] + base.source_paths()
