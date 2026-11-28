"""Q-independent tabular behavior-flow misspecification and error calibration.

Depletion and random-logit models are fixed before critic learning. Calibration
uses a fixed known-MDP reference only to choose the IQL V-noise amplitude;
the behavior model never observes the learned critic or its gradients.
"""
import numpy as np

from .mdp import roots


def behavior_model(data, kind, strength, seed):
    if kind == "none" or strength == 0:
        return data.copy()
    if kind == "deplete":
        assert 0 <= strength < 1
        model = data.copy()
        model[:, 0] *= 1 - strength
        model[:, 1:] *= ((1-model[:, 0])/(1-data[:, 0]))[:, None]
    elif kind == "logit":
        rng = np.random.default_rng(4000031 + seed)
        perturbation = rng.standard_normal(data.shape)
        logits = np.log(data) + strength * perturbation
        logits -= logits.max(-1, keepdims=True)
        model = np.exp(logits)
        model /= model.sum(-1, keepdims=True)
    else:
        raise ValueError(kind)
    assert (model > 0).all()
    np.testing.assert_allclose(model.sum(-1), 1., atol=1e-13)
    return model


def successor_weights(mdp, data):
    weights = np.zeros(len(data))
    for s in range(len(data)):
        for a in range(data.shape[1]):
            weights[mdp.successor[s, a]] += data[s, a]*mdp.continuation[s, a]**2/len(data)
    return weights


def reference_error(mdp, reference_q, data, model):
    """RMS one-step target error under the fixed replay transition law."""
    shift = roots(reference_q, model, mdp.tau)-roots(reference_q, data, mdp.tau)
    weights = successor_weights(mdp, data)
    rms = mdp.gamma*np.sqrt(np.dot(weights, shift**2))
    return float(rms), shift, weights


def matched_v_sigma(mdp, reference_q, data, epsilon, seed):
    model = behavior_model(data, "deplete", epsilon, seed)
    rms, _, weights = reference_error(mdp, reference_q, data, model)
    sigma_v = rms/(mdp.gamma*np.sqrt(weights.sum()))
    return float(sigma_v), rms
