"""Exact state-wise quartets, not OT across different observations.

OT coupling is established prior art, including OT-NFM(arXiv2604.06413)
and OptiFlow(arXiv2609.15883). This helper enumerates24permutations, not
OptiFlow's value-weighted entropic Sinkhorn algorithm. Finite-quartet MSE
regression need not reproduce the full teacher distribution.
"""
from itertools import permutations

import jax
import jax.numpy as jnp


PERMUTATIONS=tuple(permutations(range(4)))


def match(source,target):
    """Return target permutation per state for arrays[4,...batch...,action]."""
    assert source.shape==target.shape and source.shape[0]==4 and source.ndim>=2
    source,target=jax.lax.stop_gradient(source),jax.lax.stop_gradient(target)
    table=jnp.asarray(PERMUTATIONS,jnp.int32)
    choices=target[table]
    costs=jnp.mean((source[None]-choices)**2,axis=(1,-1))
    best=costs.argmin(0)
    index=jnp.moveaxis(table[best],-1,0)
    indices=jnp.broadcast_to(index[...,None],target.shape)
    matched=jnp.take_along_axis(target,indices,axis=0)
    return jax.lax.stop_gradient(matched),index,costs.min(0)
