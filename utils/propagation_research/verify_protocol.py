"""Check baseline update equivalence and nontrivial intervention invariants."""

from algorithms.uk_offline.basic import iql_jax as base
from algorithms.uk_offline.basic import iql_residual_control as research
from utils.propagation_research.diagnose_checkpoint import differentiable_adam, build_jvp_controller

import jax
import jax.numpy as jnp
import numpy as np


def main():
    cfg = base.TrainConfig(hidden_dim=32, n_hidden=2, max_timesteps=1000,
                           iql_tau=.99, discount=.999, log_wandb=False)
    device = jax.devices()[0]
    trainer, head, tx, initial = research.initialize(cfg, {}, 5, 2, device)
    rng = np.random.RandomState(7)
    batches = {"observations": jnp.asarray(rng.normal(size=(5, 16, 5)), dtype=jnp.float32),
               "next_observations": jnp.asarray(rng.normal(size=(5, 16, 5)), dtype=jnp.float32),
               "actions": jnp.asarray(rng.uniform(-1, 1, size=(5, 16, 2)), dtype=jnp.float32),
               "rewards": -jnp.ones((5, 16, 1)), "dones": jnp.zeros((5, 16, 1))}
    original = initial.core
    for i in range(5):
        original, _ = trainer._train_step(original, {k: v[i] for k, v in batches.items()})
    scanned, _ = research.make_scan(research.build_update(trainer, head, tx, {}, (-1000, 0)))(initial, batches)
    diffs = [float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
             for x, y in zip(jax.tree_util.tree_leaves(original),
                             jax.tree_util.tree_leaves(scanned.core))]
    # XLA may reassociate floating point work inside a scan. Adam can amplify
    # tiny near-zero gradient differences, so compare both state and predictions.
    for field in ("q_params", "v_params", "actor_params", "q_opt_state", "v_opt_state"):
        error = max(float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
                    for x, y in zip(jax.tree_util.tree_leaves(getattr(original, field)),
                                    jax.tree_util.tree_leaves(getattr(scanned.core, field))))
        print(field, "scan vs eager maximum difference", error, flush=True)
    assert max(diffs) < 1e-3, max(diffs)
    for module, field, arguments in (
            (trainer.q_def, "q_params", (batches["observations"][0], batches["actions"][0])),
            (trainer.v_def, "v_params", (batches["observations"][0],))):
        x = module.apply({"params": getattr(original, field)}, *arguments)
        y = module.apply({"params": getattr(scanned.core, field)}, *arguments)
        np.testing.assert_allclose(x, y, atol=1e-4, rtol=1e-4)
    a, b = np.random.RandomState(0), np.random.RandomState(0)
    assert np.array_equal(a.randint(1000000, size=(100, 256)),
                          np.stack([b.randint(1000000, size=256) for _ in range(100)]))
    for name in ("bounded", "joint", "v4_fresh_backup", "residual_target", "lagged_weights",
                 "negative_v", "bounded_v2",
                 "v2_old_backup", "fresh_backup", "bounded_fresh", "bounded_v2_old",
                 "jvp_fit_control", "jvp_residual"):
        profile = research.PROFILES[name]
        trainer, head, tx, initial = research.initialize(cfg, profile, 5, 2, device)
        test_batches = dict(batches)
        if profile.get("jvp_control"):
            test_batches["control_scale"] = jnp.full(5, .5)
        final, logs = research.make_scan(research.build_update(
            trainer, head, tx, profile, (-1000, 0)))(initial, test_batches)
        assert int(final.core.total_it) == 5
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(final))
        assert int(final.core.v_opt_state[0].count) == 5*profile.get("v_steps", 1)
        if name == "negative_v":
            values = trainer.v_def.apply({"params": final.core.v_params}, batches["observations"][0])
            assert np.all(np.asarray(values) <= 0)
        print(name, "finite parameters and correct optimizer-step count")
    # The differentiable diagnostic must not change actual Adam forward steps.
    qp = {"x": jnp.asarray([0., 2., -3., 1.])}
    import optax
    adam = optax.adam(3e-4)
    opt_state = adam.init(qp)
    for step in range(10):
        grad = {"x": jnp.asarray([0., .1+step/10, -.2, 0.])}
        expected = adam.update(grad, opt_state, qp)
        observed = differentiable_adam(grad, opt_state, 3e-4)
        for x, y in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(observed)):
            np.testing.assert_array_equal(x, y)
        qp = optax.apply_updates(qp, expected[0])
        opt_state = expected[1]
    print("PASS: differentiable Adam matches Optax forward exactly, including inactive coordinates")
    single = {k: v[0] for k, v in batches.items()}
    gains = build_jvp_controller(trainer, cfg.qf_lr, horizon=3)(final.core, single, single, jnp.asarray(.5))
    assert np.isfinite(np.asarray(gains)).all()
    print("PASS: JVP critic response is finite with inactive Adam coordinates")
    print("PASS: original vs scanned baseline maximum difference", max(diffs))
    print("PASS: chunk sampling preserves original NumPy minibatch sequence")
    values = (np.zeros((2, 4)), np.zeros((2, 4)), np.zeros(4), np.zeros(4),
              {"q_opt_state_finite_fraction": .9999999403953552, "q_opt_state_all_finite": True})
    assert research.probe_summary(values, .9, -100, 0, 10)["finite"]
    values[-1]["q_opt_state_all_finite"] = False
    assert not research.probe_summary(values, .9, -100, 0, 10)["finite"]
    print("PASS: finite decisions use boolean reductions, not a rounded float32 ratio")


if __name__ == "__main__":
    main()
