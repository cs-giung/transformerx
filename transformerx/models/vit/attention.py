"""
This module contains functions and classes for the attention module.

Classes:
    AttentionParams:
    AttentionInputs:
    AttentionConfig:

Functions:
    forward_fn: Forward function for the attention module.
"""
import math
from functools import partial
from typing import NamedTuple

import jax
from einops import einsum
from transformerx.typing import Array, ArrayLike


class AttentionConfig(NamedTuple): # pylint: disable=missing-class-docstring
    hidden_size: int
    num_attention_heads: int


class AttentionInputs(NamedTuple): # pylint: disable=missing-class-docstring
    hidden_states: ArrayLike


class AttentionParams(NamedTuple): # pylint: disable=missing-class-docstring
    q_proj_w: ArrayLike
    q_proj_b: ArrayLike
    k_proj_w: ArrayLike
    k_proj_b: ArrayLike
    v_proj_w: ArrayLike
    v_proj_b: ArrayLike
    o_proj_w: ArrayLike
    o_proj_b: ArrayLike


@partial(jax.jit, static_argnames='config')
def forward_fn(
        params: AttentionParams,
        inputs: AttentionInputs,
        config: AttentionConfig,
    ) -> Array:
    """Forward function for the attention module."""
    # pylint: disable=invalid-name, too-many-locals
    x = inputs.hidden_states

    _, _, M = x.shape
    H = config.num_attention_heads
    K = config.hidden_size // config.num_attention_heads
    V = config.hidden_size // config.num_attention_heads

    q_proj_w = params.q_proj_w.reshape(M, H, K)
    k_proj_w = params.k_proj_w.reshape(M, H, K)
    v_proj_w = params.v_proj_w.reshape(M, H, V)
    o_proj_w = params.o_proj_w.reshape(H, V, M)
    q_proj_b = params.q_proj_b.reshape(1, H, 1, K)
    k_proj_b = params.k_proj_b.reshape(1, H, 1, K)
    v_proj_b = params.v_proj_b.reshape(1, H, 1, V)
    o_proj_b = params.o_proj_b.reshape(1, 1, M)

    q = einsum(x, q_proj_w, 'B S M, M H K -> B H S K') + q_proj_b
    k = einsum(x, k_proj_w, 'B D M, M H K -> B H D K') + k_proj_b
    v = einsum(x, v_proj_w, 'B D M, M H V -> B H D V') + v_proj_b

    qk = einsum(q, k, 'B H S K, B H D K -> B H S D') / math.sqrt(K)
    qk = jax.nn.softmax(qk)

    qkv = einsum(qk, v, 'B H S D, B H D V -> B H S V')
    out = einsum(qkv, o_proj_w, 'B H S V, H V M -> B S M') + o_proj_b

    return out
