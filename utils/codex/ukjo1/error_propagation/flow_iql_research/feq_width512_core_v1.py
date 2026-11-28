"""FEQ width-512 control: unchanged N9 learner, BC Flow and MDN actor losses.

Both four-layer critic and actor/Flow architectures use width 512. The
discount is selected explicitly; no optimizer or policy-extraction changes.
"""
from pathlib import Path
from utils.codex.ukjo1.error_propagation.flow_iql_research import sample_budget_core_v1 as base

ENVIRONMENTS = base.ENVIRONMENTS
DISCOUNTS = {'g995_w512_n9': .995, 'g999_w512_n9': .999}
RECIPES = tuple(DISCOUNTS)
create = base.create
make_scan = base.make_scan
actor_metadata = base.actor_metadata
export_actor = base.export_actor
make_actor_policy = base.make_actor_policy
make_teacher_policy = base.make_teacher_policy
return_bounds = base.return_bounds
lean = base.lean
probe_summary = base.probe_summary


def config_for(environment, recipe):
    cfg = base.config_for(environment, 'n9', width=512)
    cfg['discount'] = DISCOUNTS[recipe]
    cfg['onestep_specification'].update(name=recipe, discount=DISCOUNTS[recipe])
    assert cfg['actor_hidden_dims'] == cfg['value_hidden_dims'] == (512,) * 4
    return cfg


def build_update(recipe, bounds):
    assert recipe in RECIPES
    return base.build_update('n9', bounds)


def make_probe(recipe, bounds):
    assert recipe in RECIPES
    return base.make_probe('n9', bounds)


def source_paths():
    folder = Path(__file__).resolve().parent
    return [folder / name for name in (
        'feq_width512_core_v1.py', 'train_feq_width512_v1.py',
        'evaluate_feq_width512_v1.py', 'evaluate_discount_0995_n9_v1.py',
    )] + base.source_paths()
