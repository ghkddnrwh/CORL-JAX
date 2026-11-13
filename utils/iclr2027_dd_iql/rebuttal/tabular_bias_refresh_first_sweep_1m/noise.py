"""Gaussian readout fields preserving the previous size-sweep random streams."""

import numpy as np
from common import CONFIG
from scipy.signal import lfilter

def generate_seed(noise, slot, mode):
    seed = CONFIG["seeds"][slot]
    updates = noise.shape[1]
    size = noise.shape[3]
    actions = max(4, size)
    rng = np.random.default_rng(100003 + seed)
    if mode == "nn_like_field_iid":
        for start in range(0, updates, 20000):
            end = min(start + 20000, updates)
            noise[slot, start:end] = rng.standard_normal(
                (end - start, 2, size, actions)
            )[..., :size]
        return seed
    if mode != "nn_like_field":
        raise ValueError(mode)
    # Preserve the original feature geometry, including the unused last state.
    coords = np.empty((1, size + 1, actions, 2 + actions))
    for c in range(size + 1):
        for a in range(actions):
            coords[0, c, a, :2] = (0.0, c / size)
            coords[0, c, a, 2:] = np.eye(actions)[a]
    projection = rng.normal(0.0, 1.0 / 0.35, size=(2 + actions, 63))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(63,))
    phi = np.concatenate(
        (np.full((1, size + 1, actions, 1), 0.25), np.cos(coords @ projection + phase)),
        axis=-1,
    )
    phi = phi / np.linalg.norm(phi, axis=-1, keepdims=True)
    shared = rng.standard_normal(64)
    private = rng.standard_normal((2, 64))
    latent = np.sqrt(1.0) * private + np.sqrt(0.0) * shared[None, :]
    initial = np.einsum("hwad,nd->nhwa", phi, latent)[:, 0, :size]
    carry = np.zeros((1, 2, size, actions))
    for start in range(0, updates, 20000):
        end = min(start + 20000, updates)
        length = end - start
        innovations = np.empty((length, 2, size, actions))
        offset = 1 if start == 0 else 0
        if offset:
            innovations[0] = initial
        draws = rng.standard_normal((length - offset, 3, 64))
        projected = np.einsum(
            "tnd,sad->tnsa", draws[:, 1:], phi[0, :size], optimize=True
        )
        innovations[offset:] = np.sqrt(1 - 0.99**2) * projected
        values, carry = lfilter([1.0], [1.0, -0.99], innovations, axis=0, zi=carry)
        noise[slot, start:end] = values[..., :size]
    return seed
