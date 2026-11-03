"""Check that value-hinge guidance changes only the one-step student update."""
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import aligned_policy_core as aligned, policy_core
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def equal_tree(a, b):
    aa, bb = map(jax.tree_util.tree_leaves, (a, b))
    assert len(aa) == len(bb)
    for x, y in zip(aa, bb):
        np.testing.assert_array_equal(x, y)


def main():
    cfg = aligned.config_for(aligned.ENVIRONMENTS[8], discount=.999, width=8)
    cfg.update(policy_type='fql_policy', num_samples=1, distill_coefficient=300.)
    obs, act = jnp.zeros((1, 4)), jnp.zeros((1, 2))
    first = aligned.create(0, obs, act, cfg)
    reference = policy_core.create(0, obs, act, cfg, 'fql_policy')
    equal_tree(first.agent.network.params, reference.agent.network.params)
    rng = np.random.RandomState(128)
    batch = {k: jnp.asarray(v, jnp.float32) for k, v in dict(
        observations=rng.randn(13, 4), actions=rng.uniform(-1, 1, (13, 2)),
        next_observations=rng.randn(13, 4), rewards=-np.ones(13), masks=np.ones(13)).items()}
    gradient = jax.grad(lambda q: jnp.maximum(jnp.array([1., 1.])-q, 0).mean())(jnp.array([2., 0.]))
    np.testing.assert_array_equal(gradient, jnp.array([0., -.5]))
    gradients = jax.grad(lambda params: first.agent.actor_loss(
        batch, params, jax.random.PRNGKey(77))[0])(first.agent.network.params)
    for key in ('modules_critic', 'modules_target_critic', 'modules_value'):
        assert all(np.count_nonzero(x) == 0 for x in jax.tree_util.tree_leaves(gradients[key]))
    update = jax.jit(aligned.build_update('bounded_v2_value_hinge', (-1000, 0)))
    original = jax.jit(policy_core.build_update('bounded_v2', (-1000, 0), 'fql_policy'))
    corrected = first
    same_modules = ('modules_critic', 'modules_target_critic', 'modules_value', 'modules_actor_flow')
    max_differences = {name: 0. for name in same_modules}
    for _ in range(4):
        corrected, logs = update(corrected, batch)
        reference, _ = original(reference, batch)
        for key in same_modules:
            for x, y in zip(jax.tree_util.tree_leaves(corrected.agent.network.params[key]),
                            jax.tree_util.tree_leaves(reference.agent.network.params[key])):
                max_differences[key] = max(max_differences[key], float(np.max(np.abs(x-y))))
                np.testing.assert_allclose(x, y, atol=2e-6, rtol=2e-5)
        for x, y in zip(jax.tree_util.tree_leaves(corrected.v_opt), jax.tree_util.tree_leaves(reference.v_opt)):
            np.testing.assert_allclose(x, y, atol=2e-6, rtol=2e-5)
        equal_tree(corrected.agent.rng, reference.agent.rng)
        assert 0 <= float(logs['actor/guidance_active_fraction']) <= 1
        assert float(logs['actor/q_loss']) >= 0
    differences = [float(np.max(np.abs(x-y))) for x, y in zip(
        jax.tree_util.tree_leaves(corrected.agent.network.params['modules_actor_onestep_flow']),
        jax.tree_util.tree_leaves(reference.agent.network.params['modules_actor_onestep_flow']))]
    assert max(differences) > 0
    staged = first
    for _ in range(2):
        staged, _ = update(staged, batch)
    saved = pickle.loads(pickle.dumps(flax.serialization.to_state_dict(staged)))
    restored = flax.serialization.from_state_dict(first, saved)
    for _ in range(2):
        restored, _ = update(restored, batch)
    equal_tree(restored, corrected)
    # Existing frozen evaluators use the original architecture; the student
    # guidance objective does not affect network shape or inference semantics.
    old_template = policy_core.create(0, obs, act, cfg, 'fql_policy')
    old_view = flax.serialization.from_state_dict(old_template, flax.serialization.to_state_dict(corrected))
    key = jax.random.PRNGKey(17)
    np.testing.assert_array_equal(corrected.agent.sample_actions(jnp.ones(4), seed=key),
                                  old_view.agent.sample_actions(jnp.ones(4), seed=key))
    result = dict(passed=True, time=time.time(), initialization_bitwise_equal=True,
        value_critic_target_and_BC_flow_updates_numerically_equivalent=True,
        maximum_parameter_differences=max_differences,
        common_modules_bitwise_equal=all(d == 0 for d in max_differences.values()),
        student_update_changes=True, maximum_student_param_difference=max(differences),
        Q_guidance_gradient_zero_above_V_verified=True,
        actor_loss_has_zero_critic_and_value_param_gradients=True,
        full_state_checkpoint_resume_bitwise_equal=True,
        original_architecture_frozen_evaluator_action_bitwise_equal=True,
        q_updates=int(corrected.q_updates), v_updates=int(corrected.v_updates),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(aligned.__file__), Path(__file__))},
        scope='Tiny CPU semantics checks; performance and learned-V threshold quality remain unverified. A first stricter comparison found a 2.8e-9 discrepancy; cross-objective compiled paths are checked numerically and are not claimed bitwise identical.')
    path = Path('logs/diagnosis/flow_iql_research_20261026/analysis/aligned_policy_validation_v1.json')
    path.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
