"""Numerical checks for tabular DD-IQL's losses, sequencing and delayed state."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from utils.iclr2027_dd_iql import compare_tabular_methods as grid
from utils.iclr2027_dd_iql.stress_iid_min_ddq_bias import (
    check_against_original,
    make_noise,
    run_batch,
)
from utils.iclr2027_dd_iql.tabular_dd_iql import (
    dd_iql_filter_indices,
    dd_iql_update,
    validate_dd_iql_config,
)
from utils.iclr2027_dd_iql.tabular_q_backups import clipped_q_bootstrap

class TabularDDIQLTests(unittest.TestCase):
    def test_peer_weight_is_independent_of_own_residual_sign(self):
        q, v = np.array([2.0, -2.0]), np.array([1.0, -1.0])
        q_new, v_new, weights = dd_iql_update(
            q,
            v,
            np.array([10.0, 20.0]),
            np.array([5.0, -5.0]),
            np.array([-1.0, 1.0]),
            3.0,
            False,
            0.5,
            0.2,
            0.4,
            0.8,
        )
        np.testing.assert_allclose(weights, [0.2, 0.8])
        np.testing.assert_allclose(q_new, [3.2, 1.0])
        np.testing.assert_allclose(v_new, [1.32, -2.28])
        np.testing.assert_array_equal(q, [2.0, -2.0])
        np.testing.assert_array_equal(v, [1.0, -1.0])

    def test_terminal_backup_and_symmetric_expectile(self):
        q, v, weights = dd_iql_update(
            np.array([2.0, -2.0]),
            np.zeros(2),
            np.full(2, np.nan),
            np.array([4.0, -4.0]),
            np.array([-1.0, 1.0]),
            3.0,
            True,
            0.99,
            0.2,
            0.4,
            0.5,
        )
        np.testing.assert_allclose(q, [2.2, -1.0])
        np.testing.assert_allclose(v, [0.8, -0.8])
        np.testing.assert_array_equal(weights, [0.5, 0.5])

    def test_cross_schedule_and_convex_rate_validation(self):
        np.testing.assert_array_equal(dd_iql_filter_indices(1, 12), [0])
        np.testing.assert_array_equal(dd_iql_filter_indices(2, 12), [1, 0])
        np.testing.assert_array_equal(dd_iql_filter_indices(3, 0), [1, 2, 0])
        np.testing.assert_array_equal(dd_iql_filter_indices(3, 1), [2, 0, 1])
        np.testing.assert_array_equal(dd_iql_filter_indices(3, 2), [1, 2, 0])
        for values in (
            (1.1, 0.1, 0.7, 0.1),
            (0.1, -0.1, 0.7, 0.1),
            (0.1, 0.1, np.nan, 0.1),
            (0.1, 0.1, 0.7, 1.1),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_dd_iql_config(*values)

    def test_batched_updates_against_independent_scalar_reference(self):
        rng = np.random.default_rng(19)
        seeds, steps, states = 3, 41, 3
        noise = rng.normal(size=(seeds, steps, 2, states, 4))
        actions = rng.integers(4, size=(seeds, steps))
        explore = rng.random((seeds, steps)) < 0.3
        for delay in (1, 7, steps + 1):
            for behavior in ("uniform", "epsilon_greedy"):
                with self.subTest(delay=delay, behavior=behavior):
                    result = run_batch(
                        "dd_iql",
                        noise,
                        actions,
                        0.93,
                        0.25,
                        1,
                        delay,
                        behavior,
                        explore,
                        0.82,
                        0.37,
                        0.41,
                    )
                    for seed in range(seeds):
                        q = np.zeros((2, states, 4))
                        v = np.zeros((2, states))
                        target = q.copy()
                        delayed = target + noise[seed, 0]
                        delayed_v = v.copy()
                        state = 0
                        for t in range(steps):
                            action = actions[seed, t]
                            if behavior == "epsilon_greedy" and not explore[seed, t]:
                                action = (
                                    (q[:, state] + noise[seed, t, :, state])
                                    .mean(0)
                                    .argmax()
                                )
                            successor = action % states
                            old_q, old_v, old_target = q.copy(), v.copy(), target.copy()
                            for i in range(2):
                                j = 1 - i
                                advantage = (
                                    old_target[j, state, action]
                                    + noise[seed, t, j, state, action]
                                    - old_v[j, state]
                                    if delay == 1
                                    else delayed[j, state, action] - delayed_v[j, state]
                                )
                                w = 0.82 if advantage >= 0 else 0.18
                                backup = 0.93 * old_v[i, successor]
                                q[i, state, action] = (
                                    0.75 * old_q[i, state, action] + 0.25 * backup
                                )
                                residual = (
                                    old_target[i, state, action]
                                    + noise[seed, t, i, state, action]
                                    - old_v[i, state]
                                )
                                v[i, state] = old_v[i, state] + 0.37 * w * residual
                            target = 0.59 * old_target + 0.41 * q
                            if (t + 1) % delay == 0:
                                delayed = target + noise[seed, t]
                                delayed_v = v.copy()
                            state = successor
                        for actual, expected in (
                            (result["q_final"][seed], q),
                            (result["value_final"][seed], v),
                            (result["q_target_final"][seed], target),
                            (result["value_delayed_final"][seed], delayed_v),
                            (result["q_delayed_observed_final"][seed], delayed),
                        ):
                            np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_grid_terminal_target_and_old_self_loop_value(self):
        # One terminal transition: Q learns reward immediately, V reads old Q_target.
        with patch.object(grid, "choose_action", return_value=1):
            terminal = grid.train(
                "dd_iql",
                grid_height=1,
                grid_width=2,
                num_updates=1,
                alpha=0.25,
                alpha_v=0.5,
                gamma=0.99,
                noise_sigma=0,
                goal_reward=2,
                log_every=1,
                q_target_tau=0.4,
            )
        np.testing.assert_allclose(terminal["Q_ensemble_final"][:, 0, 0, 1], 0.5)
        np.testing.assert_array_equal(terminal["value_ensemble_final"], 0)
        np.testing.assert_allclose(terminal["Q_target_final"][:, 0, 0, 1], 0.2)
        # A self-loop after a wall penalty distinguishes old V from freshly updated V.
        with patch.object(grid, "choose_action", return_value=0):
            loop = grid.train(
                "dd_iql",
                grid_height=1,
                grid_width=2,
                num_updates=2,
                alpha=0.5,
                alpha_v=0.5,
                gamma=0.9,
                noise_sigma=0,
                wall_reward=-2,
                log_every=1,
                q_target_tau=1,
                dd_delay_steps=1,
            )
        np.testing.assert_allclose(loop["Q_ensemble_final"][:, 0, 0, 0], -1.5)
        np.testing.assert_allclose(loop["value_ensemble_final"][:, 0, 0], -0.15)

    def test_grid_and_batch_match_for_replayed_noise_and_data(self):
        rng = np.random.default_rng(28)
        path = Path(grid.__file__)
        for states in (1, 2, 4):
            noise = rng.normal(size=(2, 53, 2, states, 4))
            actions = rng.integers(4, size=(2, 53))
            explore = rng.random((2, 53)) < 0.4
            for mode in ("nn_like_field_iid", "nn_like_field"):
                with self.subTest(states=states, mode=mode):
                    checks = check_against_original(
                        path.read_text(),
                        path,
                        noise,
                        actions,
                        0.97,
                        0.23,
                        dd_delay_steps=7,
                        noise_mode=mode,
                        behavior="epsilon_greedy",
                        explore_mask=explore,
                        expectile_tau=0.83,
                        alpha_v=0.31,
                        q_target_tau=0.19,
                    )
                    self.assertTrue(all(check["passed"] for check in checks))

    def test_learned_v_and_q_greedy_policy_are_saved_separately(self):
        histories = [
            grid.train(
                "dd_iql",
                grid_height=2,
                grid_width=3,
                num_updates=20,
                log_every=5,
                noise_mode="nn_like_field_iid",
                noise_sigma=0.5,
                seed=seed,
            )
            for seed in (2, 3)
        ]
        for h in histories:
            np.testing.assert_array_equal(
                h["greedy_policy_final"], h["Q_final"].argmax(-1)
            )
            max_q = h["Q_final"].max(-1)
            max_q[0, -1] = 0
            np.testing.assert_allclose(h["V"][-1], max_q)
        aggregated = grid.aggregate_seed_histories(histories, [2, 3])
        with tempfile.TemporaryDirectory() as tmp:
            grid.save_seed_aggregate("DD-IQL", aggregated, tmp, tmp)
            saved_v = np.load(Path(tmp) / "DD_IQL_value_ensemble_final_all.npy")
            np.testing.assert_array_equal(
                saved_v[0], histories[0]["value_ensemble_final"]
            )
            policy = np.load(Path(tmp) / "DD_IQL_greedy_policy_all_seeds.npy")
            np.testing.assert_array_equal(policy[0], histories[0]["greedy_policy_final"])
            self.assertIn(
                "value_bias_mean",
                (Path(tmp) / "DD_IQL_metrics_mean_std.csv").read_text(),
            )

    def test_two_action_updates_match_reference_for_every_method(self):
        rng = np.random.default_rng(81)
        path = Path(grid.__file__)
        noise = rng.normal(size=(2, 67, 2, 2, 2))
        actions = rng.integers(2, size=(2, 67))
        explore = rng.random((2, 67)) < 0.3
        for mode in ("nn_like_field_iid", "nn_like_field"):
            for behavior in ("uniform", "epsilon_greedy"):
                for dd_method in ("dd_q", "dd_iql"):
                    with self.subTest(mode=mode, behavior=behavior, dd_method=dd_method):
                        checks = check_against_original(
                            path.read_text(),
                            path,
                            noise,
                            actions,
                            0.99,
                            0.17,
                            dd_delay_steps=9,
                            noise_mode=mode,
                            behavior=behavior,
                            explore_mask=explore,
                            dd_method=dd_method,
                            expectile_tau=0.99,
                            alpha_v=0.21,
                            q_target_tau=0.08,
                        )
                        self.assertTrue(all(check["passed"] for check in checks))

    def test_two_actions_retain_the_exact_original_noise_outputs(self):
        path = Path(grid.__file__)
        for mode in ("nn_like_field_iid", "nn_like_field"):
            kwargs = dict(field_rho=0.97, field_action_scale=1.0, num_states=2)
            four = make_noise(
                path.read_text(), path, [4, 5], 15, 1.0, mode, num_actions=4, **kwargs
            )
            two = make_noise(
                path.read_text(), path, [4, 5], 15, 1.0, mode, num_actions=2, **kwargs
            )
            np.testing.assert_array_equal(two, four[..., :2])

    def test_large_action_axes_match_reference_updates(self):
        path = Path(grid.__file__)
        rng = np.random.default_rng(91)
        for size in (8, 16):
            noise = rng.normal(size=(2, 37, 2, size, size))
            actions = rng.integers(size, size=(2, 37))
            explore = rng.random((2, 37)) < 0.3
            for mode in ("nn_like_field_iid", "nn_like_field"):
                with self.subTest(size=size, mode=mode):
                    checks = check_against_original(
                        path.read_text(),
                        path,
                        noise,
                        actions,
                        0.999,
                        0.1,
                        dd_delay_steps=10,
                        noise_mode=mode,
                        behavior="epsilon_greedy",
                        explore_mask=explore,
                        expectile_tau=0.99,
                    )
                    self.assertTrue(all(c["passed"] for c in checks))

    def test_large_field_projection_matches_sequential_field(self):
        path = Path(grid.__file__)
        for size in (8, 16):
            values = make_noise(
                path.read_text(),
                path,
                [3],
                17,
                1.0,
                "nn_like_field",
                field_rho=0.97,
                field_features=64,
                field_lengthscale=0.35,
                field_action_scale=1.0,
                num_states=size,
                num_actions=size,
            )
            field = grid.NNLikeErrorField(
                n=2,
                height=1,
                width=size + 1,
                sigma=1.0,
                rho=0.97,
                feature_dim=64,
                shared_fraction=0.0,
                lengthscale=0.35,
                action_scale=1.0,
                seed=100006,
                spatial_correlation=True,
                num_actions=size,
            )
            direct = []
            for t in range(17):
                if t:
                    field.step()
                direct.append(field.full()[:, 0, :size])
            np.testing.assert_allclose(values[0], np.array(direct), atol=1e-12)

    def test_df_sd_against_scalar_paper_equations(self):
        rng = np.random.default_rng(101)
        steps, states, actions_count = 43, 4, 4
        noise = rng.normal(size=(2, steps, 2, states, actions_count))
        actions = rng.integers(actions_count, size=(2, steps))
        explore = rng.random((2, steps)) < 0.4
        for method, reduction in (
            ("df_iql", "min"), ("df_iql", "mean"), ("sd_iql", "min")
        ):
            for delay in (1, 7, steps + 1):
                for behavior in ("uniform", "epsilon_greedy"):
                    with self.subTest(method=method, delay=delay, behavior=behavior):
                        h = run_batch(
                            method,
                            noise,
                            actions,
                            0.93,
                            0.25,
                            1,
                            delay,
                            behavior,
                            explore,
                            0.82,
                            0.37,
                            0.41,
                            df_reduction=reduction,
                        )
                        for seed in range(2):
                            q = np.zeros((2, states, actions_count))
                            v = np.zeros((1 if method == "df_iql" else 2, states))
                            target = q.copy()
                            snapshot = noise[seed, 0].copy()
                            snapshot_v = v.copy()
                            state = 0
                            for t in range(steps):
                                action = actions[seed, t]
                                if behavior == "epsilon_greedy" and not explore[seed, t]:
                                    action = (
                                        (q[:, state] + noise[seed, t, :, state])
                                        .mean(0)
                                        .argmax()
                                    )
                                successor = action % states
                                old_q, old_v, old_target = (
                                    q.copy(),
                                    v.copy(),
                                    target.copy(),
                                )
                                for i in range(2):
                                    k = 0 if method == "df_iql" else i
                                    q[i, state, action] = (
                                        0.75 * old_q[i, state, action]
                                        + 0.25 * 0.93 * old_v[k, successor]
                                    )
                                for k in range(len(v)):
                                    current = (
                                        old_target[:, state, action]
                                        + noise[seed, t, :, state, action]
                                    )
                                    if method == "df_iql":
                                        regression_target = (
                                            min(current) if reduction == "min"
                                            else sum(current) / len(current)
                                        )
                                        delayed_target = (
                                            min(snapshot[:, state, action])
                                            if reduction == "min"
                                            else sum(snapshot[:, state, action]) / 2
                                        )
                                        advantage = (
                                            regression_target - old_v[0, state]
                                            if delay == 1
                                            else delayed_target - snapshot_v[0, state]
                                        )
                                    else:
                                        regression_target = current[k]
                                        advantage = (
                                            current[k] - old_v[k, state]
                                            if delay == 1
                                            else snapshot[k, state, action]
                                            - snapshot_v[k, state]
                                        )
                                    weight = 0.82 if advantage >= 0 else 0.18
                                    v[k, state] = old_v[k, state] + 0.37 * weight * (
                                        regression_target - old_v[k, state]
                                    )
                                target = 0.59 * old_target + 0.41 * q
                                if (t + 1) % delay == 0:
                                    snapshot = target + noise[seed, t]
                                    snapshot_v = v.copy()
                                state = successor
                            for key, expected in (
                                ("q_final", q),
                                ("value_final", v),
                                ("q_target_final", target),
                                ("value_delayed_final", snapshot_v),
                                ("q_delayed_observed_final", snapshot),
                            ):
                                np.testing.assert_allclose(
                                    h[key][seed], expected, atol=1e-14
                                )
                            if method == "df_iql":
                                np.testing.assert_array_equal(
                                    h["q_final"][seed, 0], h["q_final"][seed, 1]
                                )

    def test_df_sd_match_grid_reference_across_sizes_and_delays(self):
        path = Path(grid.__file__)
        rng = np.random.default_rng(102)
        for size in (2, 4, 8, 16):
            noise = rng.normal(size=(2, 39, 2, size, size))
            actions = rng.integers(size, size=(2, 39))
            explore = rng.random((2, 39)) < 0.3
            for method, reduction in (
                ("df_iql", "min"), ("df_iql", "mean"), ("sd_iql", "min")
            ):
                for delay in (1, 10):
                    with self.subTest(size=size, method=method, delay=delay):
                        checks = check_against_original(
                            path.read_text(),
                            path,
                            noise,
                            actions,
                            0.999,
                            0.1,
                            dd_delay_steps=delay,
                            dd_method=method,
                            behavior="epsilon_greedy",
                            explore_mask=explore,
                            expectile_tau=0.99,
                            df_reduction=reduction,
                        )
                        self.assertTrue(all(c["passed"] for c in checks))

    def test_d1_self_and_cross_filters_share_current_noise_observation(self):
        # Q_target stays zero. Errors reverse sign on the second update,
        # exposing accidental use of epsilon_{t-1} in the filtering weight.
        noise = np.zeros((1, 2, 2, 1, 2))
        noise[0, 0, :, 0, 0] = [1.0, -1.0]
        noise[0, 1, :, 0, 0] = [-1.0, 1.0]
        actions = np.zeros((1, 2), dtype=int)
        for method, expected in (
            ("sd_iql", [0.3775, 0.4225]),
            ("dd_iql", [-0.4225, -0.3775]),
        ):
            with self.subTest(method=method):
                h = run_batch(
                    method, noise, actions, 0.99, 0.0, 1, 1,
                    expectile_tau=0.9, alpha_v=0.5,
                )
                np.testing.assert_allclose(h["value_final"][0, :, 0], expected)

    def test_min_q_reductions_use_the_correct_critic_and_action_axes(self):
        values = np.array([[6.0, 0.0], [0.0, 2.0]])
        for reduction, expected in (
            ("max_min", 0.0),
            ("min_max", 2.0),
            ("max_mean", 3.0),
        ):
            with self.subTest(reduction=reduction):
                self.assertEqual(clipped_q_bootstrap(values, reduction), expected)
                np.testing.assert_array_equal(
                    clipped_q_bootstrap(np.stack([values, values + 10]), reduction),
                    [expected, expected + 10],
                )
        with self.assertRaises(ValueError):
            clipped_q_bootstrap(values, "unknown")

    def test_min_q_variants_match_independent_scalar_updates_and_grid(self):
        path = Path(grid.__file__)
        rng = np.random.default_rng(231)
        noise = rng.normal(size=(2, 61, 2, 4, 4))
        actions = rng.integers(4, size=(2, 61))
        for reduction in ("max_min", "min_max", "max_mean"):
            with self.subTest(reduction=reduction):
                h = run_batch(
                    "clipped_double_q",
                    noise,
                    actions,
                    0.97,
                    0.23,
                    1,
                    clipped_q_backup=reduction,
                )
                for seed in range(2):
                    q = np.zeros((2, 4, 4))
                    state = 0
                    for t in range(61):
                        action = actions[seed, t]
                        successor = action % 4
                        view = [
                            [
                                q[i, successor, a] + noise[seed, t, i, successor, a]
                                for a in range(4)
                            ]
                            for i in range(2)
                        ]
                        if reduction == "max_min":
                            target = max(
                                min(view[i][a] for i in range(2)) for a in range(4)
                            )
                        elif reduction == "min_max":
                            target = min(max(view[i]) for i in range(2))
                        else:
                            target = max(
                                sum(view[i][a] for i in range(2)) / 2 for a in range(4)
                            )
                        for i in range(2):
                            q[i, state, action] += 0.23 * (
                                0.97 * target - q[i, state, action]
                            )
                        state = successor
                    np.testing.assert_allclose(h["q_final"][seed], q, atol=1e-14)
                checks = check_against_original(
                    path.read_text(),
                    path,
                    noise,
                    actions,
                    0.97,
                    0.23,
                    clipped_q_backup=reduction,
                )
                self.assertTrue(all(check["passed"] for check in checks))

if __name__ == "__main__":
    unittest.main()
