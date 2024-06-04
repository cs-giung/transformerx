"""Functions and classes for the global response normalization module."""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from transformerx.typing import Array, ArrayLike


class GRNConfig(NamedTuple): # pylint: disable=missing-class-docstring
    grn_eps: float


class GRNInputs(NamedTuple): # pylint: disable=missing-class-docstring
    hidden_states: ArrayLike


class GRNParams(NamedTuple): # pylint: disable=missing-class-docstring
    weight: ArrayLike
    bias: ArrayLike


@partial(jax.jit, static_argnames='config')
def forward_fn(
        params: GRNParams,
        inputs: GRNInputs,
        config: GRNConfig,
    ) -> Array:
    """Forward function for performing global response normalization."""
    x = inputs.hidden_states
    y = jnp.sqrt(jnp.sum(x**2, axis=(-3, -2), keepdims=True))
    y = y / (jnp.mean(y, axis=-1, keepdims=True) + config.grn_eps)
    z = jnp.multiply(x, y)
    z = z * params.weight[None, None, None]
    z = z + params.bias[None, None, None]
    return x + z
