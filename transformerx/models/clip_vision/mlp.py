"""
This module contains functions and classes for the MLP module.

Classes:
    MLPParams:
    MLPInputs:
    MLPConfig:

Functions:
    foward_fn: Forward function for the MLP module.
"""
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
    u_proj_w: ArrayLike
    u_proj_b: ArrayLike
    d_proj_w: ArrayLike
    d_proj_b: ArrayLike


@partial(jax.jit, static_argnames='config')
def forward_fn(
        params: MLPParams,
        inputs: MLPInputs,
        config: MLPConfig, # pylint: disable=unused-argument
    ) -> Array:
    """Forward function for the MLP module."""
    x = inputs.hidden_states
    x = einsum(x, params.u_proj_w, 'B S M, M H -> B S H')
    x = x + params.u_proj_b[None, None]
    x = jnp.multiply( # pylint: disable=not-callable
        x, jax.nn.sigmoid(1.702 * x))
    x = einsum(x, params.d_proj_w, 'B S H, H M -> B S M')
    x = x + params.d_proj_b[None, None]
    return x
