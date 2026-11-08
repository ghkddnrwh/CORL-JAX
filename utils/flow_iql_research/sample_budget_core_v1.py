"""Seed0 ten-task bootstrap-count sweep; frozen best9 actor/deployment.

Only N_bootstrap changes. Antmaze uses minimum target heads, all other
tasks use their mean. Every learner also trains the unchanged MDN9 actor.
"""
from pathlib import Path
from utils.flow_iql_research import target_ablation_core_v1 as base
from utils.flow_iql_research import mixture_onestep_deploy_core_v1 as actor_core
from utils.flow_iql_research import selection_v2
import jax
import jax.numpy as jnp
import numpy as np


ENVIRONMENTS = base.ENVIRONMENTS
SAMPLE_COUNTS = (8, 9, 16, 18, 27, 32, 36)
SETTINGS = {f'n{n}': n for n in SAMPLE_COUNTS}
RECIPES = tuple(SETTINGS)
create = base.create
make_scan = base.make_scan
actor_metadata = base.actor_metadata
export_actor = base.export_actor
make_actor_policy = base.make_actor_policy
make_teacher_policy = base.make_teacher_policy
return_bounds = base.return_bounds
lean = base.lean


def reference_recipe(environment):
    assert environment in ENVIRONMENTS
    return 'ant_min_clip' if environment.startswith('antmaze-') else 'mean_clip'


def config_for(environment, recipe, width=None):
    assert recipe in SETTINGS
    reference = reference_recipe(environment)
    n = SETTINGS[recipe]
    cfg = base.config_for(environment, reference, width)
    cfg.update(backup_samples=n, num_samples=9)
    cfg['onestep_specification'].update(
        name=recipe, backup_samples=n, reference_target_recipe=reference,
        actor_label_samples=9, deployment_flow_samples=9)
    return cfg


def aggregate(heads, cfg):
    assert cfg['backup_Q_aggregation'] in ('mean', 'min')
    return heads.min(0) if cfg['backup_Q_aggregation'] == 'min' else heads.mean(0)


def base_update(recipe, bounds):
    n = SETTINGS[recipe]

    def update(state, batch):
        agent, net = state.agent, state.agent.network
        cfg = agent.config
        assert cfg['backup_samples'] == n and cfg['num_samples'] == 9
        new_rng, rng = jax.random.split(agent.rng)
        obs, noise = lean.bank(agent, batch['next_observations'], jax.random.fold_in(rng, 61723), n)
        actions = lean.flow(agent, obs, noise, 10)
        heads = net.select('target_critic')(obs, actions=actions)
        proposal_q = aggregate(heads, cfg)
        root = selection_v2.empirical_expectile(proposal_q, cfg['expectile'])
        raw_y = batch['rewards'] + cfg['discount'] * batch['masks'] * root
        y = jnp.clip(raw_y, *bounds)
        _, actor_key = jax.random.split(rng)

        def loss(params):
            q = net.select('critic')(batch['observations'], actions=batch['actions'], params=params)
            critic_loss = jnp.mean(jnp.sum((q-y)**2, axis=0))
            bc_loss, info = agent.actor_loss(batch, params, actor_key)
            return critic_loss + bc_loss, dict(
                critic_loss=critic_loss, BC_loss=bc_loss,
                **{'actor/'+k: v for k, v in info.items()})

        new_net, logs = net.apply_loss_fn(loss)
        agent.target_update(new_net, 'critic')
        agent = agent.replace(network=new_net, rng=new_rng)
        score = jnp.mean(jnp.where(proposal_q-root >= 0, cfg['expectile'], 1-cfg['expectile']) * (proposal_q-root), axis=0)
        logs.update({
            'backup/clip_fraction': jnp.mean((raw_y != y).astype(jnp.float32)),
            'backup/would_clip_fraction': jnp.mean((raw_y != y).astype(jnp.float32)),
            'backup/lower_out_of_bounds_fraction': jnp.mean((raw_y < bounds[0]).astype(jnp.float32)),
            'backup/upper_out_of_bounds_fraction': jnp.mean((raw_y > bounds[1]).astype(jnp.float32)),
            'backup/would_clip_abs_mean': jnp.mean(jnp.abs(raw_y-y)),
            'backup/finite_proposal_score_abs_mean': jnp.mean(jnp.abs(score)),
            'backup/mean_minus_min_head_value': jnp.mean(heads.mean(0)-heads.min(0)),
            'backup/proposal_count': jnp.asarray(n, jnp.float32),
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
        # This remains an old-state best9 label, including for N8 and N36.
        obs, noise, target, heads, _, _ = actor_core.teacher_batch(state, batch, rng, spec)
        actor, info = state.actor.apply_loss_fn(
            lambda p: actor_core.actor_loss(state, obs, noise, target, heads, spec, p))
        logs.update({'deployment/'+k: v for k, v in info.items()})
        return state.replace(base=learner, actor=actor, actor_updates=state.actor_updates+1), logs
    return update


def make_probe(recipe, bounds):
    n = SETTINGS[recipe]
    # Parent diagnostics operate on the same learner and actor. Their TD
    # target is overwritten below with this run's actual bootstrap count.
    previous = base.make_probe('mean_clip', bounds)

    @jax.jit
    def probe(state, batch):
        values = previous(state, batch)
        agent = state.base.agent
        cfg = agent.config
        obs, noise = lean.bank(agent, batch['next_observations'], jax.random.PRNGKey(41019), n)
        actions = lean.flow(agent, obs, noise, 10)
        heads = agent.network.select('target_critic')(obs, actions=actions)
        q = aggregate(heads, cfg)
        root = selection_v2.empirical_expectile(q, cfg['expectile'])
        mean_root = selection_v2.empirical_expectile(heads.mean(0), cfg['expectile'])
        min_root = selection_v2.empirical_expectile(heads.min(0), cfg['expectile'])
        weights = jnp.where(q-root >= 0, cfg['expectile'], 1-cfg['expectile'])
        score = jnp.mean(weights*(q-root), axis=0)
        ess = jnp.sum(weights, axis=0)**2 / jnp.sum(weights**2, axis=0)
        raw_y = batch['rewards'] + cfg['discount']*batch['masks']*root
        y = jnp.clip(raw_y, *bounds)
        finite = values['finite'] & jnp.all(jnp.isfinite(raw_y)) & jnp.all(jnp.isfinite(heads))
        values['base'].update(y=y, finite=finite)
        values.update(
            finite=finite, diagnostic_raw_target=raw_y, diagnostic_final_target=y,
            diagnostic_outside=(raw_y != y).astype(jnp.float32),
            diagnostic_mean_minus_min_root=mean_root-min_root,
            diagnostic_proposal_shape=jnp.asarray(q.shape),
            sampling_root=root, sampling_root_gap_to_max=q.max(0)-root,
            sampling_within_state_Q_std=q.std(0), sampling_expectile_ESS=ess,
            sampling_absolute_score=jnp.abs(score))
        return values
    return probe


def probe_summary(values, bounds):
    result = base.probe_summary(values, bounds)
    result['diagnostic_actual_proposal_count'] = int(np.asarray(values['diagnostic_proposal_shape'])[0])
    for key in ('sampling_root', 'sampling_root_gap_to_max', 'sampling_within_state_Q_std',
                'sampling_expectile_ESS', 'sampling_absolute_score'):
        result[key+'_mean'] = float(np.asarray(values[key], np.float64).mean())
    return result


def source_paths():
    return [Path(__file__).resolve()] + base.source_paths()
