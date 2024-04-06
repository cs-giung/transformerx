"""
RMS Normalization Module.

This module contains functions and classes for performing RMS normalization.

Classes:
    RMSNormParams: NamedTuple for RMS normalization parameters.
    RMSNormInputs: NamedTuple for RMS normalization inputs.
    RMSNormConfig: NamedTuple for RMS normalization configuration.

Functions:
    forward_fn: Forward function for RMS normalization.

"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from transformerx.typing import Array, ArrayLike


class RMSNormParams(NamedTuple): # pylint: disable=missing-class-docstring
    weight: ArrayLike


class RMSNormInputs(NamedTuple): # pylint: disable=missing-class-docstring
    inputs: ArrayLike


class RMSNormConfig(NamedTuple): # pylint: disable=missing-class-docstring
    eps: float


@partial(jax.jit, static_argnames='config')
def forward_fn(
        params: RMSNormParams,
        inputs: RMSNormInputs,
        config: RMSNormConfig,
    ) -> Array:
    """
    Forward function for RMS normalization.

    Args:
        params (RMSNormParams): parameters for RMS normalization.
        inputs (RMSNormInputs): inputs to be normalized.
        config (RMSNormConfig): configuration for RMS normalization.

    Returns:
        a normalized inputs.
    """
    x = inputs.inputs
    x = x / jnp.sqrt((x * x).mean(axis=-1, keepdims=True) + config.eps)
    x = x * params.weight
    return x
