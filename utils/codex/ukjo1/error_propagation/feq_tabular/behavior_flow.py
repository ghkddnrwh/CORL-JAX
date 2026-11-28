"""An exactly auditable one-dimensional conditional monotone behavior flow.

Action intervals have constant transition/reward and Q within each bin. The
flow transports a Gaussian through its CDF and a learned piecewise-linear
inverse behavior CDF. The categorical kernel in engine_v2 is the EXACT
interval-index distribution of this continuous-action flow, not an added
neural flow-matching claim. MLE parameters depend only on offline actions.
"""
import numpy as np
from scipy.special import ndtr

EDGES = np.linspace(-1., 1., 5)


def offline_counts(behavior, seed, per_state=1024):
    rng = np.random.default_rng(3000011+seed)
    return np.array([rng.multinomial(per_state, p) for p in behavior])


def fit(counts, smoothing=0.):
    positive = np.asarray(counts, dtype=float)+smoothing
    return positive/positive.sum(-1, keepdims=True)


def transport(probabilities, gaussian):
    u = ndtr(np.asarray(gaussian))
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.
    index = np.minimum(np.searchsorted(cdf, u, side='right'), len(probabilities)-1)
    left = np.r_[0., cdf[:-1]][index]
    fraction = (u-left)/probabilities[index]
    return EDGES[index]+np.diff(EDGES)[index]*fraction


def region(actions):
    return np.minimum(np.searchsorted(EDGES[1:], actions, side='right'), 3)
