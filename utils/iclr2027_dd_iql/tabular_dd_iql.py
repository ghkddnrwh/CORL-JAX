"""Tabular SGD counterparts of the default cross-filter DD-IQL critic losses.

Q_i bootstraps its own *old* V_i. V_i regresses to its own target Q_i, while
the asymmetric weight uses delayed Q_j - V_j from the peer j. The factor 2
and minibatch/ensemble reduction constants in squared-loss gradients are
absorbed into alpha_q and alpha_v.
"""

import numpy as np

def dd_iql_filter_indices(ensemble_size, delayed_round):
    """Match DD-IQL's cross schedule; N=1 reduces to self filtering."""
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be >= 1")
    if ensemble_size == 1:
        return np.zeros(1, dtype=int)
    shift = 1 + delayed_round % (ensemble_size - 1)
    return (np.arange(ensemble_size) + shift) % ensemble_size


def validate_dd_iql_config(alpha_q, alpha_v, expectile_tau, q_target_tau):
    for name, value in (
        ("alpha_q", alpha_q),
        ("alpha_v", alpha_v),
        ("expectile_tau", expectile_tau),
        ("q_target_tau", q_target_tau),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


def dd_iql_update(
    q_sa,
    v_s,
    v_next,
    target_q_sa,
    delayed_peer_advantage,
    reward,
    done,
    gamma,
    alpha_q,
    alpha_v,
    expectile_tau,
):
    """Return Q(s,a), V(s), and weights without mutating any old-state input.

    The final axis indexes ensemble members; optional leading axes index
    independent seeds. reward/done have only the leading axes (or are scalars).
    target_q_sa includes the chosen approximation-error observation, if any.
    No max/min is used in these learning targets. Greedy action extraction is
    a separate operation on the resulting Q tables.
    """
    weight = np.where(delayed_peer_advantage >= 0.0, expectile_tau, 1.0 - expectile_tau)
    backup = np.asarray(reward)[..., None] + gamma * np.where(
        np.asarray(done)[..., None],
        0.0,
        v_next,
    )
    q_new = (1.0 - alpha_q) * q_sa + alpha_q * backup
    rate = alpha_v * weight
    v_new = (1.0 - rate) * v_s + rate * target_q_sa
    return q_new, v_new, weight
