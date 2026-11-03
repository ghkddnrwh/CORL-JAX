"""A state-only actor with a learned categorical route and linear action heads.

Training can materialize all head targets. Deployment first chooses a route
from the hidden representation, then evaluates ONLY that linear action head.
No critic, behavior flow, ODE, or bank of realized actions is needed.
"""
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class RoutedFixedActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    components: int = 32

    @nn.compact
    def __call__(self, observations, return_components=False):
        x = observations
        for i, width in enumerate(self.hidden_dims):
            x = nn.gelu(nn.Dense(width, name=f"hidden_{i}")(x))
        logits = nn.Dense(self.components, name="route")(x)
        kernel = self.param(
            "action_kernel", nn.initializers.normal(self.hidden_dims[-1] ** -.5),
            (self.hidden_dims[-1], self.components, self.action_dim),
        )
        bias = self.param("action_bias", nn.initializers.zeros,
                          (self.components, self.action_dim))
        if return_components:
            return dict(logits=logits,
                        means=jnp.einsum("...h,hkd->...kd", x, kernel) + bias)
        route = jnp.argmax(logits, axis=-1)
        selected_kernel = jnp.moveaxis(kernel, 1, 0)[route]
        selected_bias = bias[route]
        return jnp.clip(jnp.einsum("...h,...hd->...d", x, selected_kernel)
                        + selected_bias, -1., 1.)


def make_policy(metadata, backend="cpu"):
    assert metadata["requires_critic"] is False
    assert metadata["requires_BC_flow"] is False
    assert metadata["flow_integration_steps"] == 0
    assert metadata["generated_candidate_actions"] == 0
    definition = RoutedFixedActor(tuple(metadata["hidden_dims"]),
                                  metadata["action_dim"], metadata["components"])

    def policy(params, observations, key):
        return definition.apply({"params": params}, observations)

    return jax.jit(policy, backend=backend)
