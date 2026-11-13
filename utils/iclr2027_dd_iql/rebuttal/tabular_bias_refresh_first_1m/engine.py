"""DF/SD/DD with one current filter construction per block, followed by D-1 reuses."""

import numpy as np
from numba import njit

def diagnostic_schedule(updates=1000000):
    if updates == 1000000:
        offsets = np.random.default_rng(20261112).permutation(1000)
        return np.unique(
            np.r_[1000 * np.arange(1000) + offsets + 1, 1, 100000, updates]
        ).astype(np.int64)
    return np.arange(1, updates + 1, dtype=np.int64)


@njit(cache=True, fastmath=False)
def run_seed(method, noise, actions, delay, tau, diagnostic_steps, log_every=1000):
    updates, n, states, action_count = noise.shape
    vn = 1 if method == 0 else n
    q = np.zeros((n, states, action_count))
    target = q.copy()
    value = np.zeros((vn, states))
    frozen_y = np.zeros_like(q)
    frozen_noise = np.zeros_like(q)
    frozen_v = value.copy()
    weights = np.zeros((vn, states, action_count))
    count = 1 + (updates + log_every - 1) // log_every
    bias = np.zeros(count)
    mean_q = np.zeros(count)
    rmse = np.zeros(count)
    value_curve = np.zeros((count, states))
    q100 = q.copy()
    v100 = value.copy()
    target100 = target.copy()
    samples = len(diagnostic_steps)
    dq = np.empty((samples, n, states, action_count))
    dt = np.empty_like(dq)
    de = np.empty_like(dq)
    df = np.empty_like(dq)
    dfn = np.empty_like(dq)
    dv = np.empty((samples, vn, states))
    dfv = np.empty_like(dv)
    dw = np.empty((samples, vn, states, action_count))
    used_w = np.empty((samples, vn))
    used_backup = np.empty((samples, n))
    ds = np.empty(samples, dtype=np.int64)
    da = np.empty(samples, dtype=np.int64)
    old_v = np.zeros(vn)
    next_v = np.zeros(vn)
    observed = np.zeros(n)
    state = 0
    logged = 0
    sample_index = 0
    refresh_count = 0
    for t in range(updates):
        # The first update consumes the very same readout noise as the new filter.
        if t % delay == 0:
            refresh_count += 1
            for i in range(n):
                for s in range(states):
                    for a in range(action_count):
                        frozen_y[i, s, a] = target[i, s, a] + noise[t, i, s, a]
                        frozen_noise[i, s, a] = noise[t, i, s, a]
            frozen_v[:] = value
            for i in range(vn):
                for s in range(states):
                    for a in range(action_count):
                        if method == 0:
                            advantage = (
                                min(frozen_y[0, s, a], frozen_y[1, s, a])
                                - frozen_v[0, s]
                            )
                        else:
                            source = 1 - i if method == 2 else i
                            advantage = frozen_y[source, s, a] - frozen_v[source, s]
                        weights[i, s, a] = tau if advantage >= 0.0 else 1.0 - tau
        action = actions[t]
        successor = action % states
        sample = sample_index < samples and t + 1 == diagnostic_steps[sample_index]
        if sample:
            dq[sample_index] = q
            dt[sample_index] = target
            de[sample_index] = noise[t]
            df[sample_index] = frozen_y
            dfn[sample_index] = frozen_noise
            dv[sample_index] = value
            dfv[sample_index] = frozen_v
            dw[sample_index] = weights
            ds[sample_index] = state
            da[sample_index] = action
        for i in range(vn):
            old_v[i] = value[i, state]
            next_v[i] = value[i, successor]
        for i in range(n):
            observed[i] = target[i, state, action] + noise[t, i, state, action]
        if method == 0:
            observation = min(observed[0], observed[1])
            weight = weights[0, state, action]
            backup = 0.999 * next_v[0]
            for i in range(n):
                q[i, state, action] = 0.9 * q[i, state, action] + 0.1 * backup
            rate = 0.1 * weight
            value[0, state] = (1.0 - rate) * old_v[0] + rate * observation
            if sample:
                used_w[sample_index, 0] = weight
                used_backup[sample_index, :] = backup
        else:
            for i in range(n):
                weight = weights[i, state, action]
                backup = 0.999 * next_v[i]
                q[i, state, action] = 0.9 * q[i, state, action] + 0.1 * backup
                rate = 0.1 * weight
                value[i, state] = (1.0 - rate) * old_v[i] + rate * observed[i]
                if sample:
                    used_w[sample_index, i] = weight
                    used_backup[sample_index, i] = backup
        for i in range(n):
            for s in range(states):
                for a in range(action_count):
                    target[i, s, a] = 0.995 * target[i, s, a] + 0.005 * q[i, s, a]
        if sample:
            sample_index += 1
        state = successor
        if t + 1 == 100000:
            q100 = q.copy()
            v100 = value.copy()
            target100 = target.copy()
        if (t + 1) % log_every == 0 or t + 1 == updates:
            logged += 1
            total = 0.0
            square_total = 0.0
            max_total = 0.0
            for s in range(states):
                best = -np.inf
                for a in range(action_count):
                    estimate = (q[0, s, a] + q[1, s, a]) / 2.0
                    best = max(best, estimate)
                    total += estimate
                    square_total += estimate * estimate
                max_total += best
                value_curve[logged, s] = (
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
        value_curve,
        value,
        target,
        q100,
        v100,
        target100,
        dq,
        dt,
        de,
        df,
        dfn,
        dv,
        dfv,
        dw,
        used_w,
        used_backup,
        ds,
        da,
        refresh_count,
    )


OUTPUT_KEYS = (
    "q_final",
    "v_bias",
    "mean_q_bias",
    "q_rmse",
    "value_table",
    "value_final",
    "q_target_final",
    "checkpoint_q_100k",
    "checkpoint_value_100k",
    "checkpoint_q_target_100k",
    "diagnostic_q",
    "diagnostic_target_q",
    "diagnostic_noise",
    "diagnostic_delayed_observed_q",
    "diagnostic_delayed_noise",
    "diagnostic_value",
    "diagnostic_delayed_value",
    "diagnostic_weights",
    "diagnostic_used_weight",
    "diagnostic_used_backup",
    "diagnostic_state",
    "diagnostic_action",
    "refresh_count",
)


def pack(results, steps, updates, delay, log_every=1000):
    h = {key: np.stack([r[k] for r in results]) for k, key in enumerate(OUTPUT_KEYS)}
    h["step"] = np.unique(np.r_[np.arange(0, updates + 1, log_every), updates])
    h["diagnostic_step"] = steps
    h["diagnostic_snapshot_age"] = (steps - 1) % delay
    h["diagnostic_noise_lag"] = h["diagnostic_snapshot_age"].copy()
    h["value_bias"] = h["value_table"].mean(-1)
    h["value_rmse"] = np.sqrt((h["value_table"] ** 2).mean(-1))
    h["greedy_policy_final"] = h["q_final"].mean(1).argmax(-1)
    return h
