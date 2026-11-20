"""From-scratch N9 stability interventions; frozen reference remains untouched.

DR3 is adapted to the finite expectile backup using its normalized, detached
expectile weights. Both ONLINE feature arguments receive gradients; proposal
actions, target-critic weights, and the Bellman target do not. Terminal rows
have zero DR3 penalty. This is an adaptation, not a reproduction of CQL+DR3.
"""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from utils.codex.ukjo1.error_propagation.flow_iql_research import sample_budget_core_v1 as ref
from utils.codex.ukjo1.error_propagation.flow_iql_research import mixture_onestep_deploy_core_v1 as actor_core
from utils.codex.ukjo1.error_propagation.flow_iql_research import selection_v2

ENVIRONMENTS = tuple(e for e in ref.ENVIRONMENTS if e in (
    'antmaze-large-navigate-singletask-v0',
    'humanoidmaze-medium-navigate-singletask-v0',
    'cube-double-play-singletask-v0', 'puzzle-4x4-play-singletask-v0'))
RECIPES = ('control', 'slow_target', 'critic_batch1024', 'dr3_0p001', 'dr3_0p01')
DR3 = {'dr3_0p001': .001, 'dr3_0p01': .01}
create = ref.create
make_scan = ref.make_scan
actor_metadata = ref.actor_metadata
export_actor = ref.export_actor
make_actor_policy = ref.make_actor_policy
make_teacher_policy = ref.make_teacher_policy
return_bounds = ref.return_bounds
lean = ref.lean
probe_summary = ref.probe_summary


def config_for(environment, recipe, width=None):
    assert recipe in RECIPES and environment in ENVIRONMENTS
    cfg = ref.config_for(environment, 'n9', width)
    cfg.update(critic_batch_size=1024 if recipe == 'critic_batch1024' else 256,
               dr3_coefficient=DR3.get(recipe, 0.))
    if recipe == 'slow_target':
        cfg['tau'] = .001
    cfg['onestep_specification'].update(name=recipe, stability_intervention=recipe,
        target_EMA=cfg['tau'], critic_batch_size=cfg['critic_batch_size'],
        BC_and_actor_batch_size=256, dr3_coefficient=cfg['dr3_coefficient'])
    return cfg


def critic_features(net, observations, actions, params=None):
    q, auxiliary = net.select('critic')(observations, actions=actions,
        params=params, mutable=['intermediates'])
    leaves = jax.tree_util.tree_leaves(auxiliary['intermediates'])
    assert len(leaves) == 1, 'Expected the existing MLP penultimate feature sow'
    return q, leaves[0]


def target_inputs(state, batch, recipe, rng):
    agent = state.agent
    blocks = [batch]
    if recipe == 'critic_batch1024':
        blocks += [{k: v[i] for k, v in batch['critic_extra'].items()} for i in range(3)]
    cb = {k: jnp.concatenate([b[k] for b in blocks], axis=0)
          for k in ('observations', 'actions', 'next_observations', 'rewards', 'masks')}
    banks = []
    proposal_key = jax.random.fold_in(rng, 61723)
    for i, b in enumerate(blocks):
        key = proposal_key if i == 0 else jax.random.fold_in(proposal_key, i)
        banks.append(lean.bank(agent, b['next_observations'], key, 9))
    obs = jnp.concatenate([b[0] for b in banks], axis=1)
    noise = jnp.concatenate([b[1] for b in banks], axis=1)
    actions = jax.lax.stop_gradient(lean.flow(agent, obs, noise, 10))
    heads = agent.network.select('target_critic')(obs, actions=actions)
    q = ref.aggregate(heads, agent.config)
    root = selection_v2.empirical_expectile(q, agent.config['expectile'])
    weights = jnp.where(q-root >= 0, agent.config['expectile'], 1-agent.config['expectile'])
    weights = jax.lax.stop_gradient(weights/weights.sum(0, keepdims=True))
    return cb, obs, actions, heads, q, root, weights


def build_update(recipe, bounds):
    assert recipe in RECIPES
    if recipe in ('control', 'slow_target'):
        return ref.build_update('n9', bounds)
    coefficient = DR3.get(recipe, 0.)

    def update(state, batch):
        agent, net = state.base.agent, state.base.agent.network
        cfg = agent.config
        new_rng, rng = jax.random.split(agent.rng)
        cb, obs, actions, heads, q_next, root, weights = target_inputs(state.base, batch, recipe, rng)
        raw_y = cb['rewards'] + cfg['discount']*cb['masks']*root
        y = jax.lax.stop_gradient(jnp.clip(raw_y, *bounds))
        _, actor_key = jax.random.split(rng)

        def loss(params):
            q = net.select('critic')(cb['observations'], actions=cb['actions'], params=params)
            q_loss = jnp.mean(jnp.sum((q-y)**2, axis=0))
            regularizer = jnp.asarray(0., jnp.float32)
            if coefficient:
                _, phi = critic_features(net, cb['observations'], cb['actions'], params)
                _, next_phi = critic_features(net, obs, actions, params)
                next_feature = jnp.sum(next_phi*weights[None, ..., None], axis=1)
                # Match the reference TD loss's SUM over heads, MEAN over batch.
                regularizer = jnp.mean(jnp.sum(jnp.sum(phi*next_feature, axis=-1), axis=0)*cb['masks'])
            bc_loss, info = agent.actor_loss(batch, params, actor_key)
            return q_loss+bc_loss+coefficient*regularizer, dict(
                critic_loss=q_loss, BC_loss=bc_loss, dr3_raw=regularizer,
                dr3_loss=coefficient*regularizer, **{'actor/'+k: v for k, v in info.items()})

        new_net, logs = net.apply_loss_fn(loss)
        agent.target_update(new_net, 'critic')
        agent = agent.replace(network=new_net, rng=new_rng)
        learner = state.base.replace(agent=agent, q_updates=state.base.q_updates+1)
        _, teacher_rng = jax.random.split(state.base.agent.rng)
        spec = state.base.agent.config['onestep_specification']
        actor_obs, noise, label, teacher_heads, _, _ = actor_core.teacher_batch(state, batch, teacher_rng, spec)
        actor, info = state.actor.apply_loss_fn(
            lambda p: actor_core.actor_loss(state, actor_obs, noise, label, teacher_heads, spec, p))
        logs.update({'deployment/'+k: v for k, v in info.items()})
        logs.update({'backup/clip_fraction': jnp.mean((raw_y != y).astype(jnp.float32)),
            'backup/lower_out_of_bounds_fraction': jnp.mean((raw_y < bounds[0]).astype(jnp.float32)),
            'backup/upper_out_of_bounds_fraction': jnp.mean((raw_y > bounds[1]).astype(jnp.float32)),
            'backup/mean_minus_min_head_value': jnp.mean(heads.mean(0)-heads.min(0)),
            'backup/proposal_count': jnp.asarray(9., jnp.float32),
            'critic/batch_size': jnp.asarray(cb['actions'].shape[0], jnp.float32),
            'value/steps': jnp.asarray(0., jnp.float32)})
        return state.replace(base=learner, actor=actor, actor_updates=state.actor_updates+1), logs
    return update


def make_probe(recipe, bounds):
    return ref.make_probe('n9', bounds)


@jax.jit
def fixed_candidates(state, observations):
    obs, z = lean.bank(state.base.agent, observations, jax.random.PRNGKey(713902), 9)
    return lean.flow(state.base.agent, obs, z, 10)


@jax.jit
def representation_probe(state, batch, candidates):
    agent, net = state.base.agent, state.base.agent.network
    obs = jnp.broadcast_to(batch['observations'], (9, *batch['observations'].shape))
    scores = net.select('critic')(obs, actions=candidates).min(0)
    _, phi = critic_features(net, batch['observations'], batch['actions'])
    _, rng = jax.random.split(jax.random.PRNGKey(53910))
    cb, next_obs, next_actions, _, _, _, weights = target_inputs(state.base, batch, 'control', rng)
    _, next_phi = critic_features(net, next_obs, next_actions)
    successor = jnp.sum(next_phi*weights[None, ..., None], axis=1)
    dot = jnp.sum(phi*successor, axis=-1)
    scale = jnp.linalg.norm(phi, axis=-1)*jnp.linalg.norm(successor, axis=-1)
    return dict(scores=scores, phi=phi, feature_dot=dot,
        feature_cosine=dot/jnp.maximum(scale, 1e-12),
        feature_norm=jnp.linalg.norm(phi, axis=-1))


def source_paths():
    return [Path(__file__).resolve()] + ref.source_paths()
