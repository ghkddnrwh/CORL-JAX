"""Giant 2x2: head pessimism weight and bootstrap proposal count.

q_lambda=(1-lambda)*mean(Q1,Q2)+lambda*min(Q1,Q2), lambda in{1,.5}.
The finite expectile still has tau=.9, gamma=.999, and known projection.
N_bootstrap is9 or27. Both deployment and actor best9 labels stay at9.
"""
from pathlib import Path
from utils.flow_iql_research import target_ablation_core_v1 as base
from utils.flow_iql_research import mixture_onestep_deploy_core_v1 as actor_core
from utils.flow_iql_research import selection_v2
import jax
import jax.numpy as jnp
import numpy as np


GIANT = base.GIANT
ENVIRONMENTS = (GIANT,)
SETTINGS = {'p100_n9': (1., 9), 'p050_n9': (.5, 9),
            'p100_n27': (1., 27), 'p050_n27': (.5, 27)}
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
    weight, n = SETTINGS[recipe]
    cfg = base.config_for(environment, 'ant_min_clip', width)
    cfg.update(backup_samples=n, pessimism_weight=weight,
               backup_Q_aggregation='min' if weight == 1. else 'mean_min_midpoint',
               num_samples=9)
    cfg['onestep_specification'].update(name=recipe, backup_samples=n, pessimism_weight=weight,
                                      backup_Q_aggregation=cfg['backup_Q_aggregation'],
                                      reference_target_recipe='ant_min_clip',
                                      actor_label_samples=9, deployment_flow_samples=9)
    return cfg


def aggregate_heads(heads, weight):
    # Preserve the exact old min operation in the control cell.
    if weight == 1.:
        return heads.min(0)
    return (1.-weight)*heads.mean(0)+weight*heads.min(0)


def base_update(recipe, bounds):
    weight, n = SETTINGS[recipe]
    def update(state, batch):
        agent, net = state.agent, state.agent.network
        cfg = agent.config
        new_rng, rng = jax.random.split(agent.rng)
        obs, noise = lean.bank(agent, batch['next_observations'], jax.random.fold_in(rng, 61723), n)
        actions = lean.flow(agent, obs, noise, 10)
        heads = net.select('target_critic')(obs, actions=actions)
        proposal_q = aggregate_heads(heads, weight)
        root = selection_v2.empirical_expectile(proposal_q, cfg['expectile'])
        raw_y = batch['rewards']+cfg['discount']*batch['masks']*root
        y = jnp.clip(raw_y, *bounds)
        _, actor_key = jax.random.split(rng)
        def loss(params):
            q = net.select('critic')(batch['observations'], actions=batch['actions'], params=params)
            critic_loss = jnp.mean(jnp.sum((q-y)**2, axis=0))
            bc_loss, info = agent.actor_loss(batch, params, actor_key)
            return critic_loss+bc_loss, dict(critic_loss=critic_loss, BC_loss=bc_loss,
                                           **{'actor/'+k: v for k, v in info.items()})
        new_net, logs = net.apply_loss_fn(loss)
        agent.target_update(new_net, 'critic')
        agent = agent.replace(network=new_net, rng=new_rng)
        score = jnp.mean(jnp.where(proposal_q-root >= 0, cfg['expectile'], 1-cfg['expectile'])*(proposal_q-root), axis=0)
        logs.update({
            'backup/clip_fraction': jnp.mean((raw_y != y).astype(jnp.float32)),
            'backup/would_clip_fraction': jnp.mean((raw_y != y).astype(jnp.float32)),
            'backup/lower_out_of_bounds_fraction': jnp.mean((raw_y < bounds[0]).astype(jnp.float32)),
            'backup/upper_out_of_bounds_fraction': jnp.mean((raw_y > bounds[1]).astype(jnp.float32)),
            'backup/would_clip_abs_mean': jnp.mean(jnp.abs(raw_y-y)),
            'backup/finite_proposal_score_abs_mean': jnp.mean(jnp.abs(score)),
            'backup/mean_minus_min_head_value': jnp.mean(heads.mean(0)-heads.min(0)),
            'backup/proposal_count': jnp.asarray(n, jnp.float32),
            'backup/pessimism_weight': jnp.asarray(weight, jnp.float32),
            'value/steps': jnp.asarray(0., jnp.float32),
        })
        return state.replace(agent=agent, q_updates=state.q_updates+1), logs
    return update


def build_update(recipe, bounds):
    update_base = base_update(recipe, bounds)
    def update(state, batch):
        learner, logs = update_base(state.base, batch)
        _, rng = jax.random.split(state.base.agent.rng)
        spec = state.base.agent.config['onestep_specification']
        # Keep the original nine-action actor target. For N=27 bootstrap this
        # is a separate size9 bank, rather than claiming to reuse all27 actions.
        obs, noise, target, heads, _, _ = actor_core.teacher_batch(state, batch, rng, spec)
        actor, info = state.actor.apply_loss_fn(lambda p: actor_core.actor_loss(state, obs, noise, target, heads, spec, p))
        logs.update({'deployment/'+k: v for k, v in info.items()})
        return state.replace(base=learner, actor=actor, actor_updates=state.actor_updates+1), logs
    return update


def make_probe(recipe, bounds):
    weight, n = SETTINGS[recipe]
    previous = base.make_probe('ant_min_clip', bounds)
    @jax.jit
    def probe(state, batch):
        values = previous(state, batch)
        agent = state.base.agent
        obs, noise = lean.bank(agent, batch['next_observations'], jax.random.PRNGKey(41019), n)
        actions = lean.flow(agent, obs, noise, 10)
        heads = agent.network.select('target_critic')(obs, actions=actions)
        proposal_q = aggregate_heads(heads, weight)
        root = selection_v2.empirical_expectile(proposal_q, .9)
        mean_root = selection_v2.empirical_expectile(heads.mean(0), .9)
        min_root = selection_v2.empirical_expectile(heads.min(0), .9)
        raw_y = batch['rewards']+agent.config['discount']*batch['masks']*root
        y = jnp.clip(raw_y, *bounds)
        finite = values['finite'] & jnp.all(jnp.isfinite(raw_y)) & jnp.all(jnp.isfinite(heads))
        values['base'].update(y=y, finite=finite)
        values.update(finite=finite, diagnostic_raw_target=raw_y, diagnostic_final_target=y,
                      diagnostic_outside=(raw_y != y).astype(jnp.float32),
                      diagnostic_mean_minus_min_root=mean_root-min_root,
                      diagnostic_root_above_min=root-min_root,
                      diagnostic_root_below_mean=mean_root-root,
                      diagnostic_proposal_shape=jnp.asarray(proposal_q.shape))
        return values
    return probe


def probe_summary(values, bounds):
    result = base.probe_summary(values, bounds)
    for name in ('diagnostic_root_above_min', 'diagnostic_root_below_mean'):
        result[name+'_mean'] = float(np.asarray(values[name], np.float64).mean())
    result['diagnostic_actual_proposal_count'] = int(np.asarray(values['diagnostic_proposal_shape'])[0])
    return result


def source_paths():
    return [Path(__file__).resolve()] + base.source_paths()
