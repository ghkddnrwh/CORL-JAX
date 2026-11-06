"""Independent switches for finite-mean lean target-head aggregation and clipping.

Tau=.9 finite expectile, BC Flow10, N9 and min-online-Q best9 extraction are
preserved. A separate likelihood actor never supplies bootstrap proposals or
gradients to the base learner. Both deployment methods share that base learner.
Frozen reference modules and their environment lists are not modified.
"""
from pathlib import Path

from utils.flow_iql_research import mixture_onestep_deploy_core_v1 as original
from utils.flow_iql_research import efficient_mean_core_v1 as lean
from utils.flow_iql_research import selection_v2
from utils.flow_iql_research import lazy_mixture_location_actor_v1 as location
from utils.propagation_research.reward_bounds import return_bounds as known_bounds
from utils.flow_iql_research.vendor.flax_utils import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax


GIANT = 'antmaze-giant-navigate-singletask-v0'
ENVIRONMENTS = (*lean.ENVIRONMENTS, GIANT)
RECIPES = ('mean_clip', 'mean_unclipped', 'ant_min_clip', 'ant_min_unclipped')
State = original.State
make_scan = original.make_scan
actor_metadata = original.actor_metadata
export_actor = original.export_actor
make_teacher_policy = original.make_teacher_policy


def specification(recipe):
    assert recipe in RECIPES
    spec = original.specification('mdn9_bc')
    spec.update(name=recipe, reference_training_recipe='mdn9_bc',
                backup_Q_aggregation='min' if recipe.startswith('ant_min') else 'mean',
                target_clipping=not recipe.endswith('unclipped'),
                deployment_decoder='sampled_component_location',
                base_learner='finite_mean_lean', expectile=.9, discount=.999)
    return spec


def config_for(environment, recipe, width=None):
    assert environment in ENVIRONMENTS
    assert not recipe.startswith('ant_min') or environment.startswith('antmaze-')
    # Giant uses exactly the existing Ant family architecture/settings.
    family = lean.ENVIRONMENTS[0] if environment == GIANT else environment
    cfg = original.config_for(family, 'mdn9_bc', width)
    spec = specification(recipe)
    cfg.update(onestep_specification=spec, backup_Q_aggregation=spec['backup_Q_aggregation'],
               target_clipping=spec['target_clipping'],
               backup_kind='finite_behavior_flow_target_Q_expectile_ablation')
    return cfg


def return_bounds(environment, discount, reward_min, reward_max, source='ogbench'):
    if environment == GIANT and source == 'ogbench':
        assert -1. <= reward_min <= reward_max <= 0.
        return -1./(1.-discount), 0.
    return known_bounds(environment, discount, reward_min, reward_max, source)


def create(seed, observations, actions, cfg, recipe):
    assert cfg['onestep_specification']['name'] == recipe
    learner = lean.create(seed, observations, actions, cfg, 'lean')
    metadata = actor_metadata(cfg, observations.shape[-1], actions.shape[-1])
    definition = original.actor_definition(metadata)
    latent = jnp.zeros((*observations.shape[:-1], metadata['noise_dim']))
    params = definition.init(jax.random.fold_in(jax.random.PRNGKey(seed), 99213), observations, latent)['params']
    actor = TrainState.create(definition, params, optax.adam(cfg['lr']))
    return State(learner, actor, jnp.asarray(0))


def make_actor_policy(metadata, backend='cpu'):
    return location.make_policy(metadata, 'sampled_component_location', backend)


def base_update(recipe, bounds):
    spec = specification(recipe)
    def update(state, batch):
        agent, net = state.agent, state.agent.network
        cfg = agent.config
        new_rng, rng = jax.random.split(agent.rng)
        key = jax.random.fold_in(rng, 61723)
        actions = lean.active_actions(state, batch['next_observations'], key, 'lean')
        obs = jnp.broadcast_to(batch['next_observations'], (9, *batch['next_observations'].shape))
        heads = net.select('target_critic')(obs, actions=actions)
        proposal_q = heads.min(0) if spec['backup_Q_aggregation'] == 'min' else heads.mean(0)
        root = selection_v2.empirical_expectile(proposal_q, cfg['expectile'])
        raw_y = batch['rewards'] + cfg['discount'] * batch['masks'] * root
        projected = jnp.clip(raw_y, *bounds)
        y = projected if spec['target_clipping'] else raw_y
        _, actor_key = jax.random.split(rng)
        def loss(params):
            q = net.select('critic')(batch['observations'], actions=batch['actions'], params=params)
            q_loss = jnp.mean(jnp.sum((q-y)**2, axis=0))
            bc_loss, info = agent.actor_loss(batch, params, actor_key)
            return q_loss+bc_loss, dict(critic_loss=q_loss, BC_loss=bc_loss,
                                       **{'actor/'+k: v for k, v in info.items()})
        new_net, logs = net.apply_loss_fn(loss)
        agent.target_update(new_net, 'critic')
        agent = agent.replace(network=new_net, rng=new_rng)
        score = jnp.mean(jnp.where(proposal_q-root >= 0, cfg['expectile'], 1-cfg['expectile'])*(proposal_q-root), axis=0)
        logs.update({
            'backup/clip_fraction': jnp.mean((raw_y != y).astype(jnp.float32)),
            'backup/would_clip_fraction': jnp.mean((raw_y != projected).astype(jnp.float32)),
            'backup/lower_out_of_bounds_fraction': jnp.mean((raw_y < bounds[0]).astype(jnp.float32)),
            'backup/upper_out_of_bounds_fraction': jnp.mean((raw_y > bounds[1]).astype(jnp.float32)),
            'backup/would_clip_abs_mean': jnp.mean(jnp.abs(raw_y-projected)),
            'backup/mean_minus_min_head_value': jnp.mean(heads.mean(0)-heads.min(0)),
            'backup/finite_proposal_score_abs_mean': jnp.mean(jnp.abs(score)),
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
        obs, noise, target, heads, _, _ = original.teacher_batch(state, batch, rng, spec)
        actor, info = state.actor.apply_loss_fn(lambda p: original.actor_loss(state, obs, noise, target, heads, spec, p))
        logs.update({'deployment/'+k: v for k, v in info.items()})
        return state.replace(base=learner, actor=actor, actor_updates=state.actor_updates+1), logs
    return update


def make_probe(recipe, bounds):
    spec = specification(recipe)
    actor_probe = original.make_probe('mdn9_bc', bounds)
    @jax.jit
    def probe(state, batch):
        result = actor_probe(state, batch)
        agent = state.base.agent
        q = agent.network.select('critic')(batch['observations'], actions=batch['actions'])
        target_q = agent.network.select('target_critic')(batch['observations'], actions=batch['actions'])
        ob, noise = lean.bank(agent, batch['next_observations'], jax.random.PRNGKey(41019), 9)
        actions = lean.flow(agent, ob, noise, 10)
        heads = agent.network.select('target_critic')(ob, actions=actions)
        mean_root = selection_v2.empirical_expectile(heads.mean(0), .9)
        min_root = selection_v2.empirical_expectile(heads.min(0), .9)
        root = min_root if spec['backup_Q_aggregation'] == 'min' else mean_root
        raw_y = batch['rewards']+agent.config['discount']*batch['masks']*root
        projected = jnp.clip(raw_y, *bounds)
        y = projected if spec['target_clipping'] else raw_y
        finite = result['finite'] & jnp.all(jnp.isfinite(y)) & jnp.all(jnp.isfinite(heads))
        result.update(base=dict(q=q, target_q=target_q, y=y, finite=finite), finite=finite,
                      diagnostic_mean_minus_min_root=mean_root-min_root,
                      diagnostic_raw_target=raw_y, diagnostic_final_target=y,
                      diagnostic_outside=(raw_y != projected).astype(jnp.float32))
        return result
    return probe


def probe_summary(values, bounds):
    result = original.probe_summary(values, bounds)
    for key in ('diagnostic_mean_minus_min_root', 'diagnostic_outside'):
        result[key+'_mean'] = float(np.asarray(values[key], np.float64).mean())
    raw, final = (np.asarray(values[k], np.float64) for k in ('diagnostic_raw_target', 'diagnostic_final_target'))
    result.update(raw_target_min=float(raw.min()), raw_target_max=float(raw.max()),
                  final_target_min=float(final.min()), final_target_max=float(final.max()))
    return result


def source_paths():
    paths = [Path(__file__).resolve(), Path(location.__file__).resolve()] + original.source_paths()
    return list(dict.fromkeys(paths))
