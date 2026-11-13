"""Check current-first timing, frozen gates, and independent NumPy updates."""

import unittest

import numpy as np
from common import IQL as METHODS
from engine import pack, pack_double, run_double_q, run_seed
from measure import attach_outputs

def reference(method, noise, actions, delay, tau):
    """Small NumPy oracle: build a fixed weight table before each block."""
    states = noise.shape[2]
    q = np.zeros((2, states, states))
    target = q.copy()
    value = np.zeros((1 if method == 0 else 2, states))
    state = 0
    for t, action in enumerate(actions):
        if t % delay == 0:
            source = target + noise[t]
            if method == 0:
                advantage = source.min(0, keepdims=True) - value[..., None]
            else:
                indices = [1, 0] if method == 2 else [0, 1]
                advantage = source[indices] - value[indices, :, None]
            weight = np.where(advantage >= 0, tau, 1 - tau)
        successor = action % states
        receiver = target[:, state, action] + noise[t, :, state, action]
        if method == 0:
            receiver = receiver.min(keepdims=True)
        current = value[:, state].copy()
        backup = 0.999 * value[:, successor].copy()
        q[:, state, action] = 0.9 * q[:, state, action] + 0.1 * backup
        rate = 0.1 * weight[:, state, action]
        value[:, state] = (1 - rate) * current + rate * receiver
        target = 0.995 * target + 0.005 * q
        state = successor
    return q, value, target


class TimingTests(unittest.TestCase):
    def test_sizes_and_double_q(self):
        rng = np.random.default_rng(20261112)
        for size in (2, 4, 8, 16):
            noise = rng.normal(size=(211, 2, size, size))
            actions = rng.integers(size, size=211)
            steps = np.arange(1, 212)
            for method, name in enumerate(METHODS):
                for delay in (1, 10, 100, 1000):
                    result = run_seed(method, noise, actions, delay, 0.99, steps)
                    h = pack([result], steps, 211, delay)
                    attach_outputs(h, name, delay, 0.99)
                    for got, expected in zip(
                        (result[0], result[5], result[6]),
                        reference(method, noise, actions, delay, 0.99),
                    ):
                        np.testing.assert_allclose(got, expected, atol=1e-13, rtol=0)
            result = run_double_q(noise, actions, steps)
            q = np.zeros((2, size, size))
            state = 0
            for t, action in enumerate(actions):
                y = q[:, action] + noise[t, :, action]
                selected = y.argmax(-1)
                backup = 0.999 * y[[1, 0], selected]
                q[:, state, action] += 0.1 * (backup - q[:, state, action])
                state = action
            np.testing.assert_array_equal(result[0], q)
            h = pack_double([result], steps, 211)
            attach_outputs(h, "double_q", 1, 0.9)

    def test_reference_and_phase_zero(self):
        rng = np.random.default_rng(890711)
        updates = 211
        noise = rng.normal(size=(updates, 2, 4, 4))
        actions = rng.integers(4, size=updates)
        steps = np.arange(1, updates + 1)
        for method, name in enumerate(METHODS):
            for delay in (1, 2, 10, 100, 1000):
                for tau in (0.9, 0.99):
                    with self.subTest(method=name, D=delay, tau=tau):
                        output = run_seed(method, noise, actions, delay, tau, steps)
                        h = pack([output], steps, updates, delay)
                        attach_outputs(h, name, delay, tau)
                        for got, expected in zip(
                            (output[0], output[5], output[6]),
                            reference(method, noise, actions, delay, tau),
                        ):
                            np.testing.assert_allclose(got, expected, atol=1e-13, rtol=0)
                        lag = h["diagnostic_noise_lag"]
                        np.testing.assert_array_equal(lag, np.arange(updates) % delay)
                        np.testing.assert_array_equal(
                            h["refresh_count"], (updates + delay - 1) // delay
                        )
                        for t in range(updates):
                            first = (t // delay) * delay
                            np.testing.assert_array_equal(
                                h["diagnostic_delayed_noise"][0, t], noise[first]
                            )
                            np.testing.assert_array_equal(
                                h["diagnostic_weights"][0, t],
                                h["diagnostic_weights"][0, first],
                            )
                            if lag[t] == 0:
                                np.testing.assert_array_equal(
                                    h["diagnostic_delayed_observed_q"][0, t],
                                    h["diagnostic_target_q"][0, t] + noise[t],
                                )
                                np.testing.assert_array_equal(
                                    h["diagnostic_delayed_value"][0, t],
                                    h["diagnostic_value"][0, t],
                                )

    def test_first_update_has_current_self_gate(self):
        # A positive readout at refresh must use tau immediately, then retain it
        # for the following negative readout within that same two-step block.
        noise = np.zeros((2, 2, 4, 4))
        noise[0] = 1
        noise[1] = -1
        output = run_seed(
            1, noise, np.zeros(2, dtype=np.int64), 2, 0.9, np.array([1, 2])
        )
        h = pack([output], np.array([1, 2]), 2, 2)
        np.testing.assert_array_equal(h["diagnostic_used_weight"], 0.9)
        np.testing.assert_allclose(
            h["diagnostic_value"][0, 1, :, 0], 0.09, atol=1e-15, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
