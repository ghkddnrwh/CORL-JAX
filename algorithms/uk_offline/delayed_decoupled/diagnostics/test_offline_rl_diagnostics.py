"""Dependency-light tests: python -m unittest test_offline_rl_diagnostics.py"""

import ast
import json
import importlib.util
import tempfile
import typing
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from algorithms.uk_offline.delayed_decoupled.diagnostics.diagnostics import (
    LocalMetricStore,
    atomic_save_npz,
    compute_ensemble_diagnostics,
    make_fixed_probe_numpy,
)


def synthetic_predictions(n=3, b=10000, seed=7):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, b)).astype(np.float32)
    return {
        "q": raw,
        "q_target": raw,
        "q_delayed": raw,
        "v": np.zeros_like(raw),
        "v_delayed": np.zeros_like(raw),
        "next_v": np.zeros_like(raw),
        "rewards": np.zeros((b, 1), dtype=np.float32),
        "dones": np.zeros((b, 1), dtype=np.float32),
        "discount": np.array(0.99, dtype=np.float32),
        "filter_indices": (np.arange(n) + 1) % n,
    }


class DiagnosticsTests(unittest.TestCase):
    def test_existing_ddiql_schedule_still_cycles_and_supports_self(self):
        # Run only the original pure index-scheduler function with NumPy standing
        # in for jax.numpy, so the regression test does not require JAX.
        source = Path(__file__).with_name("ddiql_diagnostics_jax.py").read_text()
        tree = ast.parse(source)
        functions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "scheduled_filter_indices"
        ]
        namespace = {"jnp": np, "Union": typing.Union}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<schedule>", "exec"), namespace)
        schedule = namespace["scheduled_filter_indices"]
        np.testing.assert_array_equal(
            schedule(3, 0, "cross", 5), np.array([1, 2, 0])
        )
        np.testing.assert_array_equal(
            schedule(3, 1, "cross", 5), np.array([2, 0, 1])
        )
        np.testing.assert_array_equal(
            schedule(3, 4, "periodic_self", 5, 1), np.array([0, 1, 2])
        )

    def test_loo_structural_correlation_and_ddiql_cross(self):
        predictions = synthetic_predictions(n=5)
        ddiql = compute_ensemble_diagnostics(
            predictions, algorithm="DDIQL", timestep=0,
            delayed_update_period=250, iql_tau=0.7, beta=3.0,
        )
        lddiql = compute_ensemble_diagnostics(
            predictions, algorithm="LDDIQL", timestep=0,
            delayed_update_period=250, iql_tau=0.7, beta=3.0,
        )
        self.assertAlmostEqual(ddiql["scalars"]["active_filter/corr_mean"], 0.0, delta=0.03)
        self.assertAlmostEqual(lddiql["scalars"]["active_filter/corr_mean"], 0.75, delta=0.03)
        self.assertGreater(
            ddiql["scalars"]["filter/active_vs_loo_sign_disagree_fraction"], 0.1
        )
        self.assertEqual(lddiql["scalars"]["filter/active_vs_loo_sign_disagree_fraction"], 0.0)

    def test_n_two_cross_equals_leave_one_out(self):
        predictions = synthetic_predictions(n=2)
        ddiql = compute_ensemble_diagnostics(
            predictions, algorithm="DDIQL", timestep=0,
            delayed_update_period=250, iql_tau=0.7, beta=3.0,
        )
        self.assertEqual(ddiql["scalars"]["filter/active_vs_loo_sign_disagree_fraction"], 0)

    def test_n_one_is_safe(self):
        result = compute_ensemble_diagnostics(
            synthetic_predictions(n=1), algorithm="LDDIQL", timestep=0,
            delayed_update_period=250, iql_tau=0.7, beta=3.0,
        )
        self.assertTrue(np.isnan(result["scalars"]["q/corr_mean"]))

    def test_jsonl_reconciliation_archives_replayed_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalMetricStore(Path(tmp))
            store.append(0, "diagnostic", {"matrix": [[1.0, np.nan]]})
            store.append(100, "train", {"q_loss": 1.2})
            store.append(300, "diagnostic", {"q_loss": 0.5})
            with store.path.open("a", encoding="utf-8") as handle:
                handle.write('{"timestep":')  # simulate an interrupted write
            reconciliation = store.reconcile(100)
            self.assertEqual(reconciliation, {
                "kept": 2, "archived_future": 1, "archived_invalid": 1
            })
            store.append(200, "train", {"q_loss": 0.7})
            records = [json.loads(line) for line in store.path.read_text().splitlines()]
            self.assertEqual([x["timestep"] for x in records], [0, 100, 200])
            self.assertIsNone(records[0]["metrics"]["matrix"][0][1])
            archived = list((Path(tmp) / "abandoned_segments").glob("*.jsonl"))
            self.assertEqual(len(archived), 2)

    def test_fixed_probe_is_deterministic_and_rng_independent(self):
        class Buffer:
            _size = 20
            _states = np.arange(20, dtype=np.float32).reshape(20, 1)
            _actions = _states.copy()
            _rewards = _states.copy()
            _next_states = _states.copy()
            _dones = np.zeros_like(_states)

        np.random.seed(42)
        before = np.random.get_state()
        p1, m1 = make_fixed_probe_numpy(Buffer(), 7, 10)
        after = np.random.get_state()
        p2, m2 = make_fixed_probe_numpy(Buffer(), 7, 10)
        self.assertEqual(before[1].tolist(), after[1].tolist())
        self.assertEqual(m1, m2)
        np.testing.assert_array_equal(p1["observations"], p2["observations"])

    def test_npz_is_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval_logs.npz"
            atomic_save_npz([{"score": 1.0}], str(path))
            atomic_save_npz([{"score": 2.0}], str(path))
            with np.load(path) as loaded:
                self.assertEqual(loaded["score"].tolist(), [2.0])
            self.assertEqual(sorted(x.name for x in Path(tmp).iterdir()), ["eval_logs.npz"])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is optional")
    def test_three_run_comparison_script(self):
        from compare_ddiql_diagnostics import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = []
            for index, name in enumerate(("rotating", "fixed", "lddiql")):
                directory = root / name
                store = LocalMetricStore(directory)
                store.save_manifest_once({
                    "config": {"ensemble_size": 3, "filter_index_update_period": -1},
                    "probe": {"probe_index_sha256": "same-index-hash"},
                })
                store.append(5000, "diagnostic", {
                    "scalars": {"q/corr_mean": 0.5 + index * 0.1, "v/corr_mean": 0.6},
                    "matrices": {},
                })
                store.append(5000, "train", {"v_grad_norm": 0.2})
                store.append(25000, "eval", {"eval/reward_mean": float(index + 10)})
                inputs.append(directory)
            output = root / "comparison"
            argv = [
                "compare_ddiql_diagnostics.py", "--ddiql", str(inputs[0]),
                "--ddiql_fixed", str(inputs[1]), "--lddiql", str(inputs[2]),
                "--output", str(output),
            ]
            with mock.patch("sys.argv", argv):
                main()
            self.assertTrue((output / "ddiql_comparison.png").is_file())
            self.assertTrue((output / "ddiql_comparison_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
