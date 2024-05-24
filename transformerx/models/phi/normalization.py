"""Functions and classes for the RMS normalization module."""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from transformerx.typing import Array, ArrayLike


class RMSNormParams(NamedTuple): # pylint: disable=missing-class-docstring
    weight: ArrayLike


class RMSNormInputs(NamedTuple): # pylint: disable=missing-class-docstring
    hidden_states: ArrayLike


class RMSNormConfig(NamedTuple): # pylint: disable=missing-class-docstring
    rms_norm_eps: float


@partial(jax.jit, static_argnames='config')
def forward_fn(
        params: RMSNormParams,
        inputs: RMSNormInputs,
        config: RMSNormConfig,
    ) -> Array:
    """Forward function for performing RMS normalization."""
    x = inputs.hidden_states
    x = x / jnp.sqrt(jnp.mean(
        jnp.square(x), axis=-1, keepdims=True) + config.rms_norm_eps)
    x = x * params.weight
    return x
