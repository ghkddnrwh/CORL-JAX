"""Oracle operator errors and coupling, reconstructed from pre-update observations."""

import numpy as np

PROXIES = (
    "oracle",
    "accumulated",
    "readout",
    "td_update",
    "td_observed",
    "temporal_observed",
    "temporal_clean",
    "loo_observed",
    "loo_clean",
)


def derive(h, method, delay, tau):
    q = h["diagnostic_q"]
    target = h["diagnostic_target_q"]
    epsilon = h["diagnostic_noise"]
    value = h["diagnostic_value"]
    seeds, samples, _, states, actions = q.shape
    row = np.arange(seeds)[:, None]
    col = np.arange(samples)[None, :]
    visited_s = h["diagnostic_state"]
    visited_a = h["diagnostic_action"]
    successor = visited_a % states
    y = target + epsilon
    frozen_y = h["diagnostic_delayed_observed_q"]
    frozen_clean = frozen_y - h["diagnostic_delayed_noise"]
    frozen_v = h["diagnostic_delayed_value"]
    age_zero = (h["diagnostic_step"] - 1) % delay == 0
    frozen_clean[:, age_zero] = target[:, age_zero]

    if method == "df_min":
        receiver = y.min(2, keepdims=True)
        receiver_clean = target.min(2, keepdims=True)
        snapshot = frozen_y.min(2, keepdims=True)
        snapshot_clean = frozen_clean.min(2, keepdims=True)
        source = snapshot
        advantage = source - frozen_v[..., None]
        online = q.mean(2, keepdims=True)
        loo_observed = y[:, :, :1] - y[:, :, 1:]
        loo_clean = target[:, :, :1] - target[:, :, 1:]
    else:
        receiver, receiver_clean = y, target
        snapshot, snapshot_clean = frozen_y, frozen_clean
        peer = [1, 0] if method == "dd_iql" else [0, 1]
        source = frozen_y[:, :, peer]
        advantage = source - frozen_v[:, :, peer, :, None]
        online = q
        loo_observed = y - y[:, :, ::-1]
        loo_clean = target - target[:, :, ::-1]
    weights = np.where(advantage >= 0, tau, 1.0 - tau)
    np.testing.assert_array_equal(weights, h["diagnostic_weights"])
    used = weights.transpose(0, 1, 3, 4, 2)[row, col, visited_s, visited_a]
    np.testing.assert_array_equal(used, h["diagnostic_used_weight"])
    backup = 0.999 * value.transpose(0, 1, 3, 2)[row, col, successor]
    if method == "df_min":
        backup = np.repeat(backup, 2, -1)
    np.testing.assert_array_equal(backup, h["diagnostic_used_backup"])
    normalized = weights / weights.sum(-1, keepdims=True)
    operator = (normalized * receiver).sum(-1)
    accumulated = (normalized * receiver_clean).sum(-1)
    readout_error = receiver - receiver_clean
    readout = (normalized * readout_error).sum(-1)
    np.testing.assert_allclose(operator, accumulated + readout, rtol=0, atol=2e-12)
    next_value = value[..., np.arange(actions) % states]
    bellman_target = 0.999 * next_value[..., None, :]
    proxies = dict(
        oracle=receiver,
        accumulated=receiver_clean,
        readout=readout_error,
        td_update=online - bellman_target,
        td_observed=receiver - bellman_target,
        temporal_observed=receiver - snapshot,
        temporal_clean=receiver_clean - snapshot_clean,
        loo_observed=loo_observed,
        loo_clean=loo_clean,
    )
    return dict(
        operator=operator,
        operator_accumulated=accumulated,
        operator_readout=readout,
        weights=weights,
        proxies=proxies,
        q=q,
        q_ensemble=q.mean(2),
        value=value,
    )


def coupling(weights, error, cross_section=False):
    """Temporal statistics at fixed receiver/state/action, then average units."""
    if cross_section:
        # Convert fixed-time, within-state/action observations into the same
        # reducer layout: seed x observations x fixed units x 1 x 1.
        seeds, samples, receivers, states, actions = weights.shape
        weights = weights.transpose(0, 3, 4, 1, 2).reshape(
            seeds, states * actions, samples * receivers, 1, 1
        )
        error = error.transpose(0, 3, 4, 1, 2).reshape(
            seeds, states * actions, samples * receivers, 1, 1
        )
    w = weights - weights.mean(1, keepdims=True)
    e = error - error.mean(1, keepdims=True)
    covariance = (w * e).mean(1)
    scale = np.sqrt((w * w).mean(1) * (e * e).mean(1))
    valid = scale > 1e-20
    correlation = np.divide(
        covariance, scale, out=np.full_like(scale, np.nan), where=valid
    )
    valid_count = valid.sum((1, 2, 3))

    def average(x):
        return np.divide(
            np.nansum(x, axis=(1, 2, 3)),
            valid_count,
            out=np.full(len(weights), np.nan),
            where=valid_count > 0,
        )

    return dict(
        abs_corr=average(np.abs(correlation)),
        signed_corr=average(correlation),
        covariance=covariance.mean((1, 2, 3)),
        valid_fraction=valid.mean((1, 2, 3)),
    )


def attach_outputs(h, method, delay, tau):
    """Keep instantaneous normalized outputs alongside their raw pre-update inputs."""
    d = derive(h, method, delay, tau)
    for field in ("operator", "operator_accumulated", "operator_readout"):
        h["diagnostic_" + field] = d[field]
    if "weights" in d:
        h["diagnostic_weights"] = d["weights"]
    h["diagnostic_snapshot_age"] = (h["diagnostic_step"] - 1) % delay


def window_rows(h, method, delay, tau, seed_ids):
    d = derive(h, method, delay, tau)
    for field in ("operator", "operator_accumulated", "operator_readout"):
        np.testing.assert_array_equal(h["diagnostic_" + field], d[field])
    if "weights" in d:
        np.testing.assert_array_equal(h["diagnostic_weights"], d["weights"])
    steps = h["diagnostic_step"]
    windows = [("all", np.ones(len(steps), bool))] + [
        (f"{start}_{start + 100000}", (steps > start) & (steps <= start + 100000))
        for start in range(0, 1000000, 100000)
    ]
    rows = []
    for name, mask in windows:
        metrics = {}
        for key in ("operator", "q", "q_ensemble", "value"):
            if d[key] is None:
                continue
            x = d[key][:, mask]
            axes = tuple(range(1, x.ndim))
            metrics[key + "_bias"] = x.mean(axes)
            metrics[key + "_rmse"] = np.sqrt((x * x).mean(axes))
            metrics[key + "_over"] = np.maximum(x, 0).mean(axes)
            metrics[key + "_under"] = np.maximum(-x, 0).mean(axes)
        for component in ("accumulated", "readout"):
            x = d["operator_" + component][:, mask]
            metrics["operator_" + component] = x.mean((1, 2, 3))
        for proxy in PROXIES:
            for cross_section in (False, True):
                if "weights" in d:
                    stats = coupling(
                        d["weights"][:, mask],
                        d["proxies"][proxy][:, mask],
                        cross_section=cross_section,
                    )
                else:
                    stats = {
                        key: np.full(len(seed_ids), np.nan)
                        for key in (
                            "abs_corr",
                            "signed_corr",
                            "covariance",
                            "valid_fraction",
                        )
                    }
                suffix = "_cross_section" if cross_section else ""
                for key, x in stats.items():
                    metrics[proxy + suffix + "_" + key] = x
        for slot, seed in enumerate(seed_ids):
            row = dict(seed=seed, window=name, diagnostic_samples=int(mask.sum()))
            for key in ("operator", "q", "q_ensemble", "value"):
                for metric in ("bias", "rmse", "over", "under"):
                    row[key + "_" + metric] = float(
                        metrics.get(key + "_" + metric, np.full(len(seed_ids), np.nan))[
                            slot
                        ]
                    )
            for key, x in metrics.items():
                row[key] = float(x[slot])
            rows.append(row)
    return rows
