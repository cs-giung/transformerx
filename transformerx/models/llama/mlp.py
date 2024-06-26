"""Functions and classes for the MLP module."""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from einops import einsum
from transformerx.typing import Array, ArrayLike


class MLPConfig(NamedTuple): # pylint: disable=missing-class-docstring
    intermediate_size: int


class MLPInputs(NamedTuple): # pylint: disable=missing-class-docstring
    hidden_states: ArrayLike


class MLPParams(NamedTuple): # pylint: disable=missing-class-docstring
    g_proj: ArrayLike
    u_proj: ArrayLike
    d_proj: ArrayLike


@partial(jax.jit, static_argnames='config')
def forward_fn(
        params: MLPParams,
        inputs: MLPInputs,
        config: MLPConfig, # pylint: disable=unused-argument
    ) -> Array:
    """Forward function for the MLP module."""
    x = inputs.hidden_states

    g = einsum(x, params.g_proj, 'B S M, M H -> B S H')
    u = einsum(x, params.u_proj, 'B S M, M H -> B S H')
    y = jnp.multiply(jax.nn.silu(g), u) # pylint: disable=not-callable
    y = einsum(y, params.d_proj, 'B S H, H M -> B S M')

    return y
