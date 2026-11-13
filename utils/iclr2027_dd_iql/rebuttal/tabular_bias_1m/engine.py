"""Strict-float compiled execution of the frozen continuing-MDP reference."""

import sys
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference"))
from stress_iid_min_ddq_bias import run_batch

METHOD_IDS = {
    "max_min": 2,
    "double_q": 1,
    "df_min": 5,
    "df_mean": 6,
    "sd_iql": 7,
    "dd_iql": 8,
}


@njit(cache=True, fastmath=False)
def run_seed(method, noise, actions, delay, tau, log_every=1000):
    updates, _, states, action_count = noise.shape
    n = 1 if method == 0 else 2
    is_iql = method >= 5
    is_df = method == 5 or method == 6
    vn = 1 if is_df else n
    q = np.zeros((n, states, action_count))
    value = np.zeros((vn, states))
    target = q.copy()
    delayed = noise[0, :n].copy()
    delayed_v = value.copy()
    q100 = q.copy()
    v100 = value.copy()
    target100 = target.copy()
    delayed100 = delayed.copy()
    delayed_v100 = delayed_v.copy()
    count = 1 + (updates + log_every - 1) // log_every
    bias = np.zeros(count)
    mean_q = np.zeros(count)
    rmse = np.zeros(count)
    values = np.zeros((count, states))
    old_v = np.zeros(vn)
    next_v = np.zeros(vn)
    observed = np.zeros(n)
    backups = np.zeros(n)
    state = 0
    logged = 0
    for t in range(updates):
        action = actions[t]
        successor = action % states
        if is_iql:
            for i in range(vn):
                old_v[i] = value[i, state]
                next_v[i] = value[i, successor]
            for i in range(n):
                observed[i] = target[i, state, action] + noise[t, i, state, action]
            if is_df:
                observation = (
                    min(observed[0], observed[1])
                    if method == 5
                    else (observed[0] + observed[1]) / 2.0
                )
                if delay == 1:
                    advantage = observation - old_v[0]
                else:
                    a = delayed[0, state, action]
                    b = delayed[1, state, action]
                    frozen = min(a, b) if method == 5 else (a + b) / 2.0
                    advantage = frozen - delayed_v[0, state]
                weight = tau if advantage >= 0.0 else 1.0 - tau
                backup = 0.999 * next_v[0]
                for i in range(n):
                    q[i, state, action] = 0.9 * q[i, state, action] + 0.1 * backup
                rate = 0.1 * weight
                value[0, state] = (1.0 - rate) * old_v[0] + rate * observation
            else:
                for i in range(n):
                    peer = 1 - i if method == 8 else i
                    advantage = (
                        observed[peer] - old_v[peer]
                        if delay == 1
                        else delayed[peer, state, action] - delayed_v[peer, state]
                    )
                    weight = tau if advantage >= 0.0 else 1.0 - tau
                    backup = 0.999 * next_v[i]
                    q[i, state, action] = 0.9 * q[i, state, action] + 0.1 * backup
                    rate = 0.1 * weight
                    value[i, state] = (1.0 - rate) * old_v[i] + rate * observed[i]
            for i in range(n):
                for s in range(states):
                    for a in range(action_count):
                        target[i, s, a] = 0.995 * target[i, s, a] + 0.005 * q[i, s, a]
            if (t + 1) % delay == 0:
                for i in range(n):
                    for s in range(states):
                        for a in range(action_count):
                            delayed[i, s, a] = target[i, s, a] + noise[t, i, s, a]
                delayed_v[:] = value
        else:
            if method == 0:
                best = -np.inf
                for a in range(action_count):
                    candidate = q[0, successor, a] + noise[t, 0, successor, a]
                    best = max(best, candidate)
                backups[0] = 0.999 * best
            elif method == 1:
                selected0 = 0
                selected1 = 0
                best0 = -np.inf
                best1 = -np.inf
                for a in range(action_count):
                    x = q[0, successor, a] + noise[t, 0, successor, a]
                    y = q[1, successor, a] + noise[t, 1, successor, a]
                    if x > best0:
                        best0 = x
                        selected0 = a
                    if y > best1:
                        best1 = y
                        selected1 = a
                backups[0] = 0.999 * (
                    q[1, successor, selected0] + noise[t, 1, successor, selected0]
                )
                backups[1] = 0.999 * (
                    q[0, successor, selected1] + noise[t, 0, successor, selected1]
                )
            else:
                best = -np.inf
                best0 = -np.inf
                best1 = -np.inf
                for a in range(action_count):
                    x = q[0, successor, a] + noise[t, 0, successor, a]
                    y = q[1, successor, a] + noise[t, 1, successor, a]
                    if method == 2:
                        best = max(best, min(x, y))
                    elif method == 3:
                        best0 = max(best0, x)
                        best1 = max(best1, y)
                    else:
                        best = max(best, (x + y) / 2.0)
                if method == 3:
                    best = min(best0, best1)
                backups[:] = 0.999 * best
            for i in range(n):
                q[i, state, action] += 0.1 * (backups[i] - q[i, state, action])
        state = successor
        if t + 1 == 100000:
            q100 = q.copy()
            v100 = value.copy()
            target100 = target.copy()
            delayed100 = delayed.copy()
            delayed_v100 = delayed_v.copy()
        if (t + 1) % log_every == 0 or t + 1 == updates:
            logged += 1
            total = 0.0
            square_total = 0.0
            max_total = 0.0
            for s in range(states):
                best = -np.inf
                for a in range(action_count):
                    estimate = q[0, s, a] if n == 1 else (q[0, s, a] + q[1, s, a]) / 2.0
                    best = max(best, estimate)
                    total += estimate
                    square_total += estimate * estimate
                max_total += best
                if is_iql:
                    values[logged, s] = (
                        value[0, s] if vn == 1 else (value[0, s] + value[1, s]) / 2.0
                    )
            bias[logged] = max_total / states
            mean_q[logged] = total / (states * action_count)
            rmse[logged] = np.sqrt(square_total / (states * action_count))
    return (
        q,
        bias,
        mean_q,
        rmse,
        values,
        value,
        target,
        delayed,
        delayed_v,
        q100,
        v100,
        target100,
        delayed100,
        delayed_v100,
    )


def run_compiled(method, noise, actions, delay=1, tau=0.9, log_every=1000):
    results = [
        run_seed(METHOD_IDS[method], noise[i], actions[i], delay, tau, log_every)
        for i in range(len(actions))
    ]
    arrays = [np.stack([r[k] for r in results]) for k in range(14)]
    steps = np.unique(
        np.r_[np.arange(0, actions.shape[1] + 1, log_every), actions.shape[1]]
    )
    h = dict(
        step=steps,
        q_final=arrays[0],
        v_bias=arrays[1],
        mean_q_bias=arrays[2],
        q_rmse=arrays[3],
        checkpoint_q_100k=arrays[9],
    )
    if METHOD_IDS[method] >= 5:
        h.update(
            value_table=arrays[4],
            value_final=arrays[5],
            q_target_final=arrays[6],
            q_delayed_observed_final=arrays[7],
            value_delayed_final=arrays[8],
            checkpoint_value_100k=arrays[10],
            checkpoint_q_target_100k=arrays[11],
            checkpoint_q_delayed_observed_100k=arrays[12],
            checkpoint_value_delayed_100k=arrays[13],
            greedy_policy_final=arrays[0].mean(axis=1).argmax(axis=-1),
        )
        h["value_bias"] = h["value_table"].mean(axis=-1)
        h["value_rmse"] = np.sqrt(np.mean(h["value_table"] ** 2, axis=-1))
    return h


def validate_compiled():
    checks = []
    for size in (2, 4, 8, 16):
        rng = np.random.default_rng(20261111 + size)
        noise = rng.normal(size=(2, 2011, 2, size, size))
        actions = rng.integers(size, size=(2, 2011))
        for method in METHOD_IDS:
            settings = (
                ((1, 0.9), (10, 0.99), (100, 0.9), (1000, 0.99))
                if METHOD_IDS[method] >= 5
                else ((1, 0.9),)
            )
            for delay, tau in settings:
                native = (
                    "df_iql"
                    if method.startswith("df_")
                    else "clipped_double_q"
                    if method in ("max_min", "min_max", "max_mean")
                    else method
                )
                reference = run_batch(
                    native,
                    noise,
                    actions,
                    0.999,
                    0.1,
                    1000,
                    dd_delay_steps=delay,
                    expectile_tau=tau,
                    alpha_v=0.1,
                    q_target_tau=0.005,
                    clipped_q_backup=method
                    if native == "clipped_double_q"
                    else "max_min",
                    df_reduction="mean" if method == "df_mean" else "min",
                )
                actual = run_compiled(method, noise, actions, delay, tau)
                max_error = 0.0
                for key in reference:
                    np.testing.assert_allclose(
                        actual[key], reference[key], atol=1e-11, rtol=0
                    )
                    max_error = max(
                        max_error, float(np.max(np.abs(actual[key] - reference[key])))
                    )
                checks.append(
                    dict(
                        size=size,
                        method=method,
                        D=delay,
                        tau=tau,
                        max_abs_error=max_error,
                        passed=True,
                    )
                )
    return checks
