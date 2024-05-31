"""
This module contains functions and classes for performing layer normalization.

Classes:
    LayerNormParams:
    LayerNormInputs:
    LayerNormConfig:

Functions:
    forward_fn: Forward function for layer normalization.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from transformerx.typing import Array, ArrayLike


class LayerNormConfig(NamedTuple): # pylint: disable=missing-class-docstring
    layer_norm_eps: float


class LayerNormInputs(NamedTuple): # pylint: disable=missing-class-docstring
    hidden_states: ArrayLike


class LayerNormParams(NamedTuple): # pylint: disable=missing-class-docstring
    weight: ArrayLike
    bias: ArrayLike


@partial(jax.jit, static_argnames='config')
def forward_fn(
        params: LayerNormParams,
        inputs: LayerNormInputs,
        config: LayerNormConfig,
    ) -> Array:
    """Forwafd function for performing layer normalization."""
    x = inputs.hidden_states
    x = x - jnp.mean(x, axis=-1, keepdims=True)
    x = x / jnp.sqrt(
        jnp.var(x, axis=-1, keepdims=True) + config.layer_norm_eps)
    x = x * params.weight[None, None]
    x = x + params.bias[None, None]
    return x
