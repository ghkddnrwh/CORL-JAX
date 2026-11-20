"""Finite-mean lean with a direct max over nine mean-target-Q proposals.

The tau->1 endpoint is implemented explicitly. Passing tau=1 into the finite
expectile solver would include a zero denominator and does not define a unique
unconstrained asymmetric-loss minimizer. We choose the finite-sample maximum.
This is an ablation of bootstrap aggregation only; deployment remains min-Q
best9 and the actor is the original unweighted behavior flow.
"""
from pathlib import Path

from utils.flow_iql_research import efficient_mean_core_v1 as lean
from utils.flow_iql_research import selection_v2
import jax
import jax.numpy as jnp
import numpy as np


ENVIRONMENTS = lean.ENVIRONMENTS
VARIANTS = ('max_q',)
make_scan = lean.make_scan
policy_parameters = lean.policy_parameters


def config_for(environment, variant, width=None):
    assert variant == 'max_q'
    cfg = lean.config_for(environment, 'lean', width)
    cfg.update(expectile=1., backup_kind='finite_behavior_flow_mean_target_Q_max',
               backup_Q_aggregation='mean', backup_samples=9,
               bootstrap_action_aggregation='max')
    return cfg


def specification(variant):
    assert variant == 'max_q'
    return dict(variant=variant, base='finite_mean_lean', learned_V_fits=0,
                expectile_endpoint=1., endpoint_implementation='direct_max',
                backup_samples=9, flow_steps=10, target_Q_aggregation='mean',
                bootstrap_action_aggregation='max', deployment='flow_best9',
                deployment_Q_aggregation='min', extra_deployment_actor=False)


def set_guidance_config(cfg, environment, variant):
    assert variant == 'max_q'
    return cfg


def create(seed, observations, actions, cfg, variant):
    assert variant == 'max_q' and cfg['expectile'] == 1.
    return lean.create(seed, observations, actions, cfg, 'lean')


def build_update(variant, interval):
    assert variant == 'max_q'

    def update(state, batch):
        agent, net = state.agent, state.agent.network
        cfg = agent.config
        new_rng, rng = jax.random.split(agent.rng)
        proposal_key = jax.random.fold_in(rng, 61723)
        actions = lean.active_actions(state, batch['next_observations'], proposal_key, 'lean')
        obs = jnp.broadcast_to(batch['next_observations'], (9, *batch['next_observations'].shape))
        proposal_q = net.select('target_critic')(obs, actions=actions).mean(0)
        root = jnp.max(proposal_q, axis=0)
        unbounded = batch['rewards'] + cfg['discount'] * batch['masks'] * root
        target = jnp.clip(unbounded, *interval)
        _, actor_key = jax.random.split(rng)

        def loss(parameters):
            q = net.select('critic')(batch['observations'], actions=batch['actions'], params=parameters)
            critic_loss = jnp.mean(jnp.sum((q-target)**2, axis=0))
            actor_loss, actor_info = agent.actor_loss(batch, parameters, actor_key)
            return critic_loss+actor_loss, dict(critic_loss=critic_loss, BC_loss=actor_loss,
                                               **{'actor/'+k: v for k, v in actor_info.items()})

        new_net, logs = net.apply_loss_fn(loss)
        agent.target_update(new_net, 'critic')
        agent = agent.replace(network=new_net, rng=new_rng)
        # The tau=1 score being zero is an identity, not a true residual claim.
        score = jnp.mean(jnp.maximum(proposal_q-root, 0), axis=0)
        logs.update({
            'backup/clip_fraction': jnp.mean((unbounded != target).astype(jnp.float32)),
            'backup/lower_clip_fraction': jnp.mean((unbounded < interval[0]).astype(jnp.float32)),
            'backup/upper_clip_fraction': jnp.mean((unbounded > interval[1]).astype(jnp.float32)),
            'backup/finite_proposal_score_abs_mean': jnp.mean(jnp.abs(score)),
            'backup/proposal_max_mean': root.mean(),
            'backup/proposal_range_mean': jnp.mean(root-proposal_q.min(0)),
            'value/steps': jnp.asarray(0., jnp.float32),
        })
        return state.replace(agent=agent, q_updates=state.q_updates+1), logs

    return update


def make_policy(template, variant, mode='primary', backend='cpu'):
    assert variant == 'max_q'
    return lean.make_policy(template, 'lean', mode, backend)


def make_probe(variant, interval):
    assert variant == 'max_q'

    @jax.jit
    def probe(state, batch):
        agent = state.agent
        q = agent.network.select('critic')(batch['observations'], actions=batch['actions'])
        target_q = agent.network.select('target_critic')(batch['observations'], actions=batch['actions'])
        key = jax.random.PRNGKey(41019)
        obs, noise = lean.bank(agent, batch['next_observations'], key, 9)
        actions = lean.flow(agent, obs, noise, 10)
        heads = agent.network.select('target_critic')(obs, actions=actions)
        proposals = heads.mean(0)
        root = proposals.max(0)
        expectile09 = selection_v2.empirical_expectile(proposals, .9)
        sorted_q = jnp.sort(proposals, axis=0)
        unbounded = batch['rewards'] + agent.config['discount'] * batch['masks'] * root
        y = jnp.clip(unbounded, *interval)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(state)]))
        finite = finite & jnp.all(jnp.isfinite(proposals)) & jnp.all(jnp.isfinite(y))
        return dict(q=q, target_q=target_q, y=y, finite=finite,
                    max_minus_expectile09=root-expectile09,
                    max_minus_mean=root-proposals.mean(0),
                    top_two_gap=sorted_q[-1]-sorted_q[-2],
                    proposal_head_disagreement=jnp.abs(heads[0]-heads[1]),
                    unbounded_y=unbounded, proposal_q=proposals)
    return probe


def probe_summary(values, interval):
    result = lean.probe_summary(values, interval)
    for name in ('max_minus_expectile09', 'max_minus_mean', 'top_two_gap', 'proposal_head_disagreement'):
        a = np.asarray(values[name], np.float64)
        result[name+'_mean'] = float(a.mean())
        result[name+'_rms'] = float(np.sqrt(np.mean(a*a)))
    proposals = np.asarray(values['proposal_q'])
    result.update(proposal_Q_min=float(proposals.min()), proposal_Q_max=float(proposals.max()),
                  endpoint_max_not_population_error_estimate=True)
    return result


def source_paths():
    return [Path(__file__).resolve()] + lean.source_paths()
