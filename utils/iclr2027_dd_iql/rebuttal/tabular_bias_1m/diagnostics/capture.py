"""Observe the existing strict-float training loop without replacing its updates."""

import inspect
import sys
from pathlib import Path

import numpy as np
from numba import njit

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
import engine

DIAGNOSTIC_KEYS = (
    "diagnostic_q",
    "diagnostic_target_q",
    "diagnostic_noise",
    "diagnostic_delayed_observed_q",
    "diagnostic_delayed_noise",
    "diagnostic_value",
    "diagnostic_delayed_value",
    "diagnostic_used_weight",
    "diagnostic_used_backup",
    "diagnostic_state",
    "diagnostic_action",
)


def instrumented_source():
    """Insert observations into the original function; preserve every update line."""
    source = inspect.getsource(engine.run_seed.py_func)

    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)

    replace("@njit(cache=True, fastmath=False)", "@njit(cache=False, fastmath=False)")
    replace(
        "def run_seed(method, noise, actions, delay, tau, log_every=1000):",
        "def run_seed_diagnostics(method, noise, actions, delay, tau, "
        "diagnostic_steps, log_every=1000):",
    )
    replace(
        "    logged = 0\n    for t in range(updates):",
        """    logged = 0
    diagnostic_count = len(diagnostic_steps)
    dq = np.empty((diagnostic_count, n, states, action_count))
    dt = np.empty_like(dq)
    de = np.empty_like(dq)
    dd = np.empty_like(dq)
    dne = np.empty_like(dq)
    dv = np.empty((diagnostic_count, vn, states))
    ddv = np.empty_like(dv)
    dw = np.full((diagnostic_count, vn), np.nan)
    db = np.full((diagnostic_count, n), np.nan)
    ds = np.empty(diagnostic_count, dtype=np.int64)
    da = np.empty(diagnostic_count, dtype=np.int64)
    diagnostic_index = 0
    for t in range(updates):""",
    )
    replace(
        "        successor = action % states\n        if is_iql:",
        """        successor = action % states
        sample = (
            diagnostic_index < diagnostic_count
            and t + 1 == diagnostic_steps[diagnostic_index]
        )
        if sample:
            dq[diagnostic_index] = q
            dt[diagnostic_index] = target
            de[diagnostic_index] = noise[t, :n]
            dd[diagnostic_index] = delayed
            refresh_noise_index = 0 if t < delay else (t // delay) * delay - 1
            dne[diagnostic_index] = noise[refresh_noise_index, :n]
            dv[diagnostic_index] = value
            ddv[diagnostic_index] = delayed_v
            ds[diagnostic_index] = state
            da[diagnostic_index] = action
        if is_iql:""",
    )
    for indent, index in ((16, "0"), (20, "i")):
        prefix = " " * indent
        old = "\n" + prefix + "weight = tau if advantage >= 0.0 else 1.0 - tau\n"
        replace(
            old,
            old
            + prefix
            + "if sample:\n"
            + prefix
            + f"    dw[diagnostic_index, {index}] = weight\n",
        )
    replace(
        "                backup = 0.999 * next_v[0]\n",
        "                backup = 0.999 * next_v[0]\n"
        "                if sample:\n"
        "                    db[diagnostic_index, :] = backup\n",
    )
    replace(
        "                    backup = 0.999 * next_v[i]\n",
        "                    backup = 0.999 * next_v[i]\n"
        "                    if sample:\n"
        "                        db[diagnostic_index, i] = backup\n",
    )
    replace(
        "            for i in range(n):\n"
        "                q[i, state, action] += 0.1 * "
        "(backups[i] - q[i, state, action])",
        "            if sample:\n"
        "                db[diagnostic_index] = backups\n"
        "            for i in range(n):\n"
        "                q[i, state, action] += 0.1 * "
        "(backups[i] - q[i, state, action])",
    )
    replace(
        "        state = successor\n",
        "        if sample:\n"
        "            diagnostic_index += 1\n"
        "        state = successor\n",
    )
    assert source.endswith("    )\n")
    source = (
        source[:-6] + "        dq, dt, de, dd, dne, dv, ddv, dw, db, ds, da,\n    )\n"
    )
    return source


SOURCE = instrumented_source()
namespace = {"np": np, "njit": njit, "__name__": __name__}
exec(compile(SOURCE, str(Path(__file__).resolve()), "exec"), namespace)
run_seed_diagnostics = namespace["run_seed_diagnostics"]


def diagnostic_schedule(updates=1000000):
    """Balance snapshot ages without consuming the training random streams."""
    if updates == 1000000:
        offsets = np.random.default_rng(20261112).permutation(1000)
        steps = 1000 * np.arange(1000) + offsets + 1
        return np.unique(np.r_[steps, 1, 100000, updates]).astype(np.int64)
    return np.unique(np.r_[np.arange(1, updates + 1, 97), updates]).astype(np.int64)


def pack(results, steps):
    """Use the original result names, with separate diagnostic arrays."""
    a = [np.stack([r[k] for r in results]) for k in range(25)]
    updates = int(steps[-1])
    h = dict(
        step=np.unique(np.r_[np.arange(0, updates + 1, 1000), updates]),
        q_final=a[0],
        v_bias=a[1],
        mean_q_bias=a[2],
        q_rmse=a[3],
        value_table=a[4],
        value_final=a[5],
        q_target_final=a[6],
        q_delayed_observed_final=a[7],
        value_delayed_final=a[8],
        checkpoint_q_100k=a[9],
        checkpoint_value_100k=a[10],
        checkpoint_q_target_100k=a[11],
        checkpoint_q_delayed_observed_100k=a[12],
        checkpoint_value_delayed_100k=a[13],
        diagnostic_step=steps,
    )
    h["value_bias"] = h["value_table"].mean(-1)
    h["value_rmse"] = np.sqrt((h["value_table"] ** 2).mean(-1))
    h["greedy_policy_final"] = h["q_final"].mean(1).argmax(-1)
    h.update(zip(DIAGNOSTIC_KEYS, a[14:]))
    return h


def validate_capture():
    """Check identical training and pre-update snapshots, including refreshes."""
    rng = np.random.default_rng(771901)
    noise = rng.normal(size=(2011, 2, 4, 4))
    actions = rng.integers(4, size=2011)
    steps = np.array([1, 10, 11, 1000, 1001, 2011])
    checks = []
    for method in ("max_min", "df_min", "sd_iql", "dd_iql"):
        settings = (
            [(1, 0.9)]
            if method == "max_min"
            else [(d, tau) for d in (1, 10, 100, 1000) for tau in (0.9, 0.99)]
        )
        for delay, tau in settings:
            method_id = engine.METHOD_IDS[method]
            actual = run_seed_diagnostics(method_id, noise, actions, delay, tau, steps)
            expected = engine.run_seed(method_id, noise, actions, delay, tau)
            for i in range(14):
                np.testing.assert_array_equal(actual[i], expected[i])
            for sample, step in enumerate(steps):
                if step == 1:
                    np.testing.assert_array_equal(actual[14][sample], 0.0)
                    continue
                previous = engine.run_seed(
                    method_id, noise[: step - 1], actions[: step - 1], delay, tau
                )
                for capture_index, reference_index in (
                    (14, 0),
                    (15, 6),
                    (17, 7),
                    (19, 5),
                    (20, 8),
                ):
                    np.testing.assert_array_equal(
                        actual[capture_index][sample], previous[reference_index]
                    )
            checks.append(
                dict(
                    method=method,
                    D=delay,
                    tau=tau,
                    bitwise_training_match=True,
                    pre_update_snapshots_match=True,
                )
            )
    return checks
