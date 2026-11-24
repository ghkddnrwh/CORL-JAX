"""Default-task N9 sweep changing only the target-critic EMA coefficient."""
from pathlib import Path
from utils.codex.ukjo1.error_propagation.flow_iql_research import sample_budget_core_v1 as base
from utils.codex.ukjo1.error_propagation.flow_iql_research import nondefault_n9_core_v1 as catalog

FAMILIES = catalog.FAMILIES
ENVIRONMENTS = tuple(family for _, family, _ in FAMILIES)
RATES = {'ema0p001': .001, 'ema0p0005': .0005}
RECIPES = tuple(RATES)
create = base.create
make_scan = base.make_scan
actor_metadata = base.actor_metadata
export_actor = base.export_actor
make_actor_policy = base.make_actor_policy
make_teacher_policy = base.make_teacher_policy
probe_summary = base.probe_summary
lean = base.lean
return_bounds = base.return_bounds


def task_info(environment):
    code, family, default = next(t for t in FAMILIES if t[1] == environment)
    return dict(code=code, family=family, environment=environment, default_task=default, task_id=default)


def dataset_name(environment):
    assert environment in ENVIRONMENTS
    return environment.replace('-singletask', '')


def config_for(environment, recipe, width=None):
    assert environment in ENVIRONMENTS and recipe in RECIPES
    cfg = base.config_for(environment, 'n9', width)
    cfg['tau'] = RATES[recipe]
    cfg['onestep_specification'].update(name=recipe, target_EMA=cfg['tau'])
    return cfg


def build_update(recipe, bounds):
    assert recipe in RECIPES
    return base.build_update('n9', bounds)


def make_probe(recipe, bounds):
    assert recipe in RECIPES
    return base.make_probe('n9', bounds)


def source_paths():
    return [Path(__file__).resolve(), Path(catalog.__file__).resolve()] + base.source_paths()
