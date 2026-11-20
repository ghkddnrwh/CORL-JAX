"""Giant-only 2x2 discount/expectile study, preserving min-Q and projection.

The base algorithm and likelihood actor are the frozen target-ablation code.
Only discount and finite-expectile level change. Bounds track the discount.
"""
from pathlib import Path
from utils.flow_iql_research import target_ablation_core_v1 as base
from utils.flow_iql_research import selection_v2
import jax
import jax.numpy as jnp
import numpy as np


GIANT = base.GIANT
ENVIRONMENTS = (GIANT,)
SETTINGS = {'g999_t90': (.999, .9), 'g995_t90': (.995, .9),
            'g999_t95': (.999, .95), 'g995_t95': (.995, .95)}
RECIPES = tuple(SETTINGS)
create = base.create
make_scan = base.make_scan
actor_metadata = base.actor_metadata
export_actor = base.export_actor
make_actor_policy = base.make_actor_policy
make_teacher_policy = base.make_teacher_policy
return_bounds = base.return_bounds
lean = base.lean


def config_for(environment, recipe, width=None):
    assert environment == GIANT and recipe in SETTINGS
    discount, expectile = SETTINGS[recipe]
    cfg = base.config_for(environment, 'ant_min_clip', width)
    cfg.update(discount=discount, expectile=expectile)
    cfg['onestep_specification'].update(name=recipe, discount=discount, expectile=expectile,
                                      giant_search='discount_expectile_2x2',
                                      reference_target_recipe='ant_min_clip')
    return cfg


def build_update(recipe, bounds):
    assert recipe in SETTINGS
    # Frozen base update already reads both scalars from the agent config.
    return base.build_update('ant_min_clip', bounds)


def make_probe(recipe, bounds):
    assert recipe in SETTINGS
    previous = base.make_probe('ant_min_clip', bounds)
    @jax.jit
    def probe(state, batch):
        values = previous(state, batch)
        agent = state.base.agent
        obs, noise = lean.bank(agent, batch['next_observations'], jax.random.PRNGKey(41019), 9)
        actions = lean.flow(agent, obs, noise, 10)
        heads = agent.network.select('target_critic')(obs, actions=actions)
        min_values = heads.min(0)
        root = selection_v2.empirical_expectile(min_values, agent.config['expectile'])
        mean_root = selection_v2.empirical_expectile(heads.mean(0), agent.config['expectile'])
        root90 = selection_v2.empirical_expectile(min_values, .9)
        root95 = selection_v2.empirical_expectile(min_values, .95)
        raw_y = batch['rewards']+agent.config['discount']*batch['masks']*root
        y = jnp.clip(raw_y, *bounds)
        finite = values['finite'] & jnp.all(jnp.isfinite(raw_y))
        values['base'].update(y=y, finite=finite)
        values.update(finite=finite, diagnostic_mean_minus_min_root=mean_root-root,
                      diagnostic_raw_target=raw_y, diagnostic_final_target=y,
                      diagnostic_outside=(raw_y != y).astype(jnp.float32),
                      diagnostic_expectile95_minus90=root95-root90)
        return values
    return probe


def probe_summary(values, bounds):
    result = base.probe_summary(values, bounds)
    a = np.asarray(values['diagnostic_expectile95_minus90'], np.float64)
    result['diagnostic_expectile95_minus90_mean'] = float(a.mean())
    return result


def source_paths():
    return [Path(__file__).resolve()] + base.source_paths()
