"""Ten-task gamma=.995 replication of the frozen N9 learner and actor.

Only the discount and its derived return bounds change. No critic averaging,
learning-rate intervention, or deployment modification is enabled.
"""
from pathlib import Path
from utils.codex.ukjo1.error_propagation.flow_iql_research import sample_budget_core_v1 as base

ENVIRONMENTS = base.ENVIRONMENTS
RECIPES = ('g995_n9',)
create = base.create
make_scan = base.make_scan
actor_metadata = base.actor_metadata
export_actor = base.export_actor
make_actor_policy = base.make_actor_policy
make_teacher_policy = base.make_teacher_policy
return_bounds = base.return_bounds
lean = base.lean


def config_for(environment, recipe, width=None):
    assert recipe in RECIPES
    cfg = base.config_for(environment, 'n9', width)
    cfg['discount'] = .995
    cfg['onestep_specification'].update(name=recipe, discount=.995)
    return cfg


def build_update(recipe, bounds):
    assert recipe in RECIPES
    return base.build_update('n9', bounds)


def make_probe(recipe, bounds):
    assert recipe in RECIPES
    return base.make_probe('n9', bounds)


probe_summary = base.probe_summary


def source_paths():
    return [Path(__file__).resolve()] + base.source_paths()
