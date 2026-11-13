"""Gaussian readout fields with the same random streams as the previous S=A=4 run."""

import numpy as np
from common import CONFIG
from scipy.signal import lfilter

def generate_seed(noise, slot, mode):
    seed = CONFIG["seeds"][slot]
    updates = noise.shape[1]
    rng = np.random.default_rng(100003 + seed)
    if mode == "nn_like_field_iid":
        for start in range(0, updates, 20000):
            end = min(start + 20000, updates)
            noise[slot, start:end] = rng.standard_normal((end - start, 2, 4, 4))
        return seed
    if mode != "nn_like_field":
        raise ValueError(mode)
    # Preserve the original 1x5 feature geometry, including its unused fifth state.
    coords = np.empty((1, 5, 4, 6))
    for c in range(5):
        for a in range(4):
            coords[0, c, a, :2] = (0.0, c / 4)
            coords[0, c, a, 2:] = np.eye(4)[a]
    projection = rng.normal(0.0, 1.0 / 0.35, size=(6, 63))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(63,))
    phi = np.concatenate(
        (np.full((1, 5, 4, 1), 0.25), np.cos(coords @ projection + phase)), axis=-1
    )
    phi = phi / np.linalg.norm(phi, axis=-1, keepdims=True)
    shared = rng.standard_normal(64)
    private = rng.standard_normal((2, 64))
    latent = np.sqrt(1.0) * private + np.sqrt(0.0) * shared[None, :]
    initial = np.einsum("hwad,nd->nhwa", phi, latent)[:, 0, :4]
    carry = np.zeros((1, 2, 4, 4))
    for start in range(0, updates, 20000):
        end = min(start + 20000, updates)
        length = end - start
        innovations = np.empty((length, 2, 4, 4))
        offset = 1 if start == 0 else 0
        if offset:
            innovations[0] = initial
        draws = rng.standard_normal((length - offset, 3, 64))
        projected = np.einsum("tnd,sad->tnsa", draws[:, 1:], phi[0, :4], optimize=True)
        innovations[offset:] = np.sqrt(1 - 0.99**2) * projected
        values, carry = lfilter([1.0], [1.0, -0.99], innovations, axis=0, zi=carry)
        noise[slot, start:end] = values
    return seed
