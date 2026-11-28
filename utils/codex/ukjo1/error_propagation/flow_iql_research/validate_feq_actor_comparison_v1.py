"""Independent loss/gradient, learner isolation, and deployment checks."""
import argparse
import json
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from utils.codex.ukjo1.error_propagation.flow_iql_research import feq_actor_comparison_v1 as core
from utils.codex.ukjo1.error_propagation.flow_iql_research.default_gamma_pair_n9_v1 import atomic


def compare(a, b, exact=False):
    xs, ys = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(xs) == len(ys)
    largest = 0.
    for x, y in zip(xs, ys):
        x, y = np.asarray(x), np.asarray(y)
        if exact:
            np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=2e-6)
        if x.size:
            largest = max(largest, float(np.max(np.abs(x.astype(float)-y.astype(float)))))
    return largest


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    env = 'antmaze-large-navigate-singletask-v0'
    cfg = core.config_for(env, 'g995_n9', width=32)
    specs = {'fql': dict(method='fql', alpha=10.), 'cvae': dict(method='cvae', alpha=2.5)}
    r = np.random.RandomState(23)
    batch = dict(observations=jnp.asarray(r.normal(size=(8, 5)), jnp.float32),
                 next_observations=jnp.asarray(r.normal(size=(8, 5)), jnp.float32),
                 actions=jnp.asarray(r.uniform(-.8, .8, size=(8, 2)), jnp.float32),
                 rewards=jnp.full(8, -.5), masks=jnp.ones(8))
    state = core.create(0, batch['observations'][:1], batch['actions'][:1], cfg, specs)
    old = core.reference.create(0, batch['observations'][:1], batch['actions'][:1], cfg, 'g995_n9')
    compare(state.base, old.base, exact=True)
    critic = state.base.agent.network.select('critic')
    z = jnp.asarray(r.normal(size=(8, 2)), jnp.float32)
    target = core.reference.lean.flow(state.base.agent, batch['observations'], z)
    actor = state.actors['fql']
    def manual_fql(params):
        raw = actor(batch['observations'], z, params=params)
        return 10*jnp.mean(jnp.square(raw-jax.lax.stop_gradient(target))) - jnp.mean(critic(batch['observations'], actions=jnp.clip(raw, -1, 1)))
    actual = lambda params: core.fql_loss(actor, params, critic, batch['observations'], z, target, 10.)[0]
    compare(jax.value_and_grad(actual)(actor.params), jax.value_and_grad(manual_fql)(actor.params), exact=True)
    cvae = state.actors['cvae']
    post = jnp.asarray(r.normal(size=(8, 4)), jnp.float32)
    prior = jnp.asarray(r.normal(size=(8, 4)), jnp.float32)
    def manual_cvae(params):
        rec, mean, log_std, act = cvae(batch['observations'], batch['actions'], post, prior, params=params)
        q = critic(batch['observations'], actions=act)[0]
        scale = jax.lax.stop_gradient(jnp.maximum(jnp.abs(q).mean(), 1e-6))
        return jnp.square(rec-batch['actions']).mean() + .25*(mean**2+jnp.exp(2*log_std)-1-2*log_std).mean() - 2.5*q.mean()/scale
    actual_cv = lambda params: core.cvae_loss(cvae, params, critic, batch, post, prior, 2.5)[0]
    compare(jax.value_and_grad(actual_cv)(cvae.params), jax.value_and_grad(manual_cvae)(cvae.params), exact=True)
    # Q guidance has a nonzero decoder gradient, but no encoder gradient.
    def guidance_only(params):
        act = cvae(batch['observations'], prior, params=params, method='decode')
        return -critic(batch['observations'], actions=act)[0].mean()
    grads = jax.grad(guidance_only)(cvae.params)
    assert all(np.all(np.asarray(x) == 0) for x in jax.tree_util.tree_leaves(grads['encoder']))
    assert any(np.any(np.asarray(x) != 0) for x in jax.tree_util.tree_leaves(grads['decoder']))
    new_update = jax.jit(core.build_update(specs, (-200., 0.)))
    old_update = jax.jit(core.reference.build_update('g995_n9', (-200., 0.)))
    # Removing the MDN and adding two different actors must leave Q, targets,
    # BC Flow, their optimizers and PRNG trajectory unchanged.
    largest = 0.
    for _ in range(20):
        state, _ = new_update(state, batch)
        old, _ = old_update(old, batch)
        largest = max(largest, compare(state.base, old.base))
    restored = flax.serialization.from_state_dict(state, flax.serialization.to_state_dict(jax.device_get(state)))
    next_a, _ = new_update(state, batch); next_b, _ = new_update(restored, batch)
    compare(next_a, next_b, exact=True)
    for name, spec in specs.items():
        meta = core.actor_metadata(cfg, 5, 2, **spec)
        export = core.export_actor(state, name, meta)
        key = jax.random.PRNGKey(8)
        noise_key, _ = jax.random.split(key)
        latent = jax.random.normal(noise_key, (8, meta['noise_dim']))
        model = state.actors[name]
        direct = jnp.clip(model(batch['observations'], latent), -1, 1) if name == 'fql' else model(batch['observations'], latent, method='decode')
        deployed = core.make_actor_policy(meta)(export['actor_params'], batch['observations'], key)
        compare(direct, deployed)
        if name == 'cvae':
            assert set(export['actor_params']) == {'decoder'}
    atomic(args.output, dict(passed=True, base_trajectory_20_updates_max_abs_difference=largest,
        loss_and_gradient_checks=['FQL unnormalized mean-Q + same-noise flow MSE', 'CVAE posterior MSE + KL + detached normalized prior Q1'],
        resume_exact=True, deployment_parity=True, encoder_excluded_from_deployment=True,
        actor_cannot_update_base=True, devices=[str(d) for d in jax.devices()]))
    print(json.dumps(dict(passed=True, base_difference=largest)), flush=True)


if __name__ == '__main__':
    main()
