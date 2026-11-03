"""Exact best-of-N weights for a uniform ordered subset of an iid bank.

This is a sampling identity, not a change from best9 to best18. Conditional
on M candidates, average over every ordered N-subset and first-maximum tie
break. Exchangeability gives the original iid best-of-N marginal law.
"""
from itertools import permutations
import math

import numpy as np


def choose(n, k):
    return math.comb(n, k) if n >= k else 0


def weights_numpy(q, n=9):
    """q has shape (M,...); return the conditional winner probabilities."""
    q = np.asarray(q)
    assert np.isfinite(q).all() and 1 <= n <= len(q)
    # [i,j,...] compares candidate j against candidate i.
    lower = (q[None, ...] < q[:, None, ...]).sum(axis=1)
    tied = (q[None, ...] == q[:, None, ...]).sum(axis=1)
    lookup = np.asarray([choose(k, n) for k in range(len(q) + 1)], np.float64)
    result = (lookup[lower + tied] - lookup[lower]) / tied / choose(len(q), n)
    assert (result >= 0).all()
    assert np.allclose(result.sum(axis=0), 1., rtol=0., atol=1e-12)
    return result


def weights_jax(q, n=9):
    """Same fixed-bank identity, including exact ties, suitable for JIT."""
    import jax
    import jax.numpy as jnp
    assert 1 <= n <= q.shape[0]
    lower = jnp.sum(q[None, ...] < q[:, None, ...], axis=1)
    tied = jnp.sum(q[None, ...] == q[:, None, ...], axis=1)
    lookup = jnp.asarray([choose(k, n) for k in range(q.shape[0] + 1)], dtype=q.dtype)
    return jax.lax.stop_gradient((lookup[lower + tied] - lookup[lower]) / tied / choose(q.shape[0], n))


def exact_checks():
    cases = []
    for q in ([0.,1.,2.,3.], [0.,0.,1.,1.], [1.,1.,1.,1.], [3.,1.,1.,2.]):
        q = np.asarray(q)
        exact = np.zeros(4)
        ordered = list(permutations(range(4), 2))
        for subset in ordered:
            index = subset[int(np.argmax(q[list(subset)]))]
            exact[index] += 1 / len(ordered)
        found = weights_numpy(q, 2)
        assert np.allclose(found, exact, atol=1e-15, rtol=0.)
        cases.append(dict(q=q.tolist(), probabilities=found.tolist(), ordered_subsets=len(ordered)))
    best18 = weights_numpy(np.arange(18.), 9)
    assert best18[-1] == .5
    assert np.array_equal(weights_numpy(np.arange(9.), 9), np.eye(9)[-1])
    # On every ordered sample of four iid draws from this discrete reference,
    # integrating the subset weights exactly equals the direct best2 law.
    from itertools import product
    reference = np.asarray([.2,.3,.5])
    marginal = np.zeros(3)
    for bank in product(range(3), repeat=4):
        probability = float(np.prod(reference[list(bank)]))
        for index, weight in enumerate(weights_numpy(np.asarray(bank, float), 2)):
            marginal[bank[index]] += probability * weight
    direct = np.diff(np.r_[0., np.cumsum(reference)**2])
    assert np.allclose(marginal, direct, atol=1e-14, rtol=0.)
    return dict(status='pass', toy_cases=cases,
                best9_from18_unique_rank_probabilities=best18.tolist(),
                effective_teacher_rank_sample_size=float(1 / np.sum(best18**2)),
                exact_discrete_marginal=marginal.tolist(), direct_best2_marginal=direct.tolist(),
                student_noise_must_be_independent_of_teacher_bank=True,
                theoretical_scope='Conditional expectation over auxiliary ordered subset; same bestN marginal objective and no larger teacher-sampling variance at a fixed student. No optimization or RL performance guarantee.')
