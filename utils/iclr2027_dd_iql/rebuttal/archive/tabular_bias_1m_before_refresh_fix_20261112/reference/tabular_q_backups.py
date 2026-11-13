"""Critic/action reductions for the clipped Q-learning backup ablations."""

import numpy as np

CLIPPED_Q_BACKUPS = ("max_min", "min_max", "max_mean")


def clipped_q_bootstrap(candidates, reduction="max_min"):
    """Reduce [..., critic, action/candidate] to one bootstrap per batch item.

    Virtual action candidates, if present, must be flattened with the real
    action axis before calling. No discount or reward is applied here.
    """
    candidates = np.asarray(candidates)
    if candidates.ndim < 2 or min(candidates.shape[-2:]) < 1:
        raise ValueError("Need nonempty critic and action/candidate axes")
    if reduction == "max_min":
        return candidates.min(axis=-2).max(axis=-1)
    if reduction == "min_max":
        return candidates.max(axis=-1).min(axis=-1)
    if reduction == "max_mean":
        return candidates.mean(axis=-2).max(axis=-1)
    raise ValueError(f"reduction must be one of {CLIPPED_Q_BACKUPS}")
