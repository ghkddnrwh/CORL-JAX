"""Q-only late learning-rate interventions for the frozen sample-budget learner.

The old Adam moments, replay stream, BC Flow and actor schedule are retained.
Scale Adam *updates*, not gradients (Adam would largely cancel gradient scaling).
Target-Q EMA keeps the reference implementation's old-online-Q timing.
"""
from pathlib import Path

from utils.codex.ukjo1.error_propagation.flow_iql_research import sample_budget_core_v1 as reference

import jax
import jax.numpy as jnp
import optax

METHODS = ('control', 'q_lr_0p1', 'q_cosine')
START, END = 800000, 1000000


def lr_multiplier(method, completed_updates):
    assert method in METHODS
    if method == 'control':
        return jnp.asarray(1., jnp.float32)
    if method == 'q_lr_0p1':
        return jnp.asarray(.1, jnp.float32)
    # First resumed update uses 1; the update ending at END uses zero.
    progress = jnp.clip((completed_updates - START) / (END - START - 1), 0., 1.)
    return .5 * (1. + jnp.cos(jnp.pi * progress))


def scheduled_adam_step(net, loss, multiplier):
    grads, logs = jax.grad(loss, has_aux=True)(net.params)
    # Match TrainState.apply_loss_fn's diagnostics and optimizer exactly.
    maxima = [jnp.max(x) for x in jax.tree_util.tree_leaves(grads)]
    minima = [jnp.min(x) for x in jax.tree_util.tree_leaves(grads)]
    norms = [jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(grads)]
    logs.update({'grad/max': jnp.max(jnp.stack(maxima)),
                 'grad/min': jnp.min(jnp.stack(minima)),
                 'grad/norm': jnp.linalg.norm(jnp.stack(norms), ord=1)})
    updates, opt_state = net.tx.update(grads, net.opt_state, net.params)
    updates = dict(updates)
    updates['modules_critic'] = jax.tree_util.tree_map(
        lambda x: multiplier * x, updates['modules_critic'])
    params = optax.apply_updates(net.params, updates)
    return net.replace(step=net.step + 1, params=params, opt_state=opt_state), logs


def build_update(recipe, bounds, method):
    assert method in METHODS
    n = reference.SETTINGS[recipe]

    def update(state, batch):
        base = state.base
        agent, net = base.agent, base.agent.network
        cfg = agent.config
        new_rng, rng = jax.random.split(agent.rng)
        obs, noise = reference.lean.bank(agent, batch['next_observations'],
                                        jax.random.fold_in(rng, 61723), n)
        actions = reference.lean.flow(agent, obs, noise, 10)
        heads = net.select('target_critic')(obs, actions=actions)
        proposal_q = reference.aggregate(heads, cfg)
        root = reference.selection_v2.empirical_expectile(proposal_q, cfg['expectile'])
        raw_y = batch['rewards'] + cfg['discount'] * batch['masks'] * root
        y = jnp.clip(raw_y, *bounds)
        _, actor_key = jax.random.split(rng)

        def loss(params):
            q = net.select('critic')(batch['observations'], actions=batch['actions'], params=params)
            critic_loss = jnp.mean(jnp.sum((q-y)**2, axis=0))
            bc_loss, info = agent.actor_loss(batch, params, actor_key)
            return critic_loss + bc_loss, dict(critic_loss=critic_loss, BC_loss=bc_loss,
                **{'actor/'+k: v for k, v in info.items()})

        multiplier = lr_multiplier(method, base.q_updates)
        new_net, logs = scheduled_adam_step(net, loss, multiplier)
        agent.target_update(new_net, 'critic')
        learner = base.replace(agent=agent.replace(network=new_net, rng=new_rng),
                               q_updates=base.q_updates + 1)
        score = jnp.mean(jnp.where(proposal_q-root >= 0, cfg['expectile'],
                                  1-cfg['expectile']) * (proposal_q-root), axis=0)
        logs.update({
            'backup/clip_fraction': jnp.mean((raw_y != y).astype(jnp.float32)),
            'backup/proposal_count': jnp.asarray(n, jnp.float32),
            'backup/finite_proposal_score_abs_mean': jnp.mean(jnp.abs(score)),
            'value/steps': jnp.asarray(0., jnp.float32),
            'stabilization/q_lr_multiplier': multiplier,
            'stabilization/q_learning_rate': multiplier * cfg['lr'],
        })
        # Same old-state best9 labels and independent actor Adam as reference.
        spec = cfg['onestep_specification']
        label_obs, label_noise, target, label_heads, _, _ = reference.actor_core.teacher_batch(
            state, batch, rng, spec)
        actor, info = state.actor.apply_loss_fn(lambda p: reference.actor_core.actor_loss(
            state, label_obs, label_noise, target, label_heads, spec, p))
        logs.update({'deployment/'+k: v for k, v in info.items()})
        return state.replace(base=learner, actor=actor, actor_updates=state.actor_updates+1), logs

    return update


def source_paths():
    return [Path(__file__).resolve()] + reference.source_paths()
