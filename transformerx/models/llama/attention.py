"""Functions and classes for the attention module."""
import math
from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from einops import einsum, rearrange, repeat
from transformerx.typing import Array, ArrayLike


class AttentionConfig(NamedTuple): # pylint: disable=missing-class-docstring
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    sliding_window: int


class AttentionInputs(NamedTuple): # pylint: disable=missing-class-docstring
    hidden_states: ArrayLike
    attention_mask: ArrayLike
    position_ids: ArrayLike
    rope_cos: ArrayLike
    rope_sin: ArrayLike


class AttentionParams(NamedTuple): # pylint: disable=missing-class-docstring
    q_proj: ArrayLike
    k_proj: ArrayLike
    v_proj: ArrayLike
    o_proj: ArrayLike


def apply_rotary_embedding(
        arr: ArrayLike,
        cos: ArrayLike,
        sin: ArrayLike,
    ) -> Array:
    """Apply rotary embedding to an input array."""
    brr = rearrange(arr, '... (i x) -> ... i x', i=2)
    brr = brr[..., ::-1, :]
    brr = brr.at[..., 0, :].multiply(-1)
    brr = rearrange(brr, '... i x -> ... (i x)')
    a = einsum(arr, cos, 'B ... L K, B L K -> B ... L K')
    b = einsum(brr, sin, 'B ... L K, B L K -> B ... L K')
    return a + b


@partial(jax.jit, static_argnames='config')
def forward_fn(
        params: AttentionParams,
        inputs: AttentionInputs,
        config: AttentionConfig,
    ) -> Array:
    """Forward function for the attention module."""
    # pylint: disable=invalid-name,too-many-locals
    x = inputs.hidden_states

    _, _, M = x.shape
    H = config.num_key_value_heads
    R = config.num_attention_heads // config.num_key_value_heads
    K = config.hidden_size // config.num_attention_heads
    V = config.hidden_size // config.num_attention_heads

    q_proj = params.q_proj.reshape(M, H, R, K).transpose(0, 2, 1, 3)
    k_proj = params.k_proj.reshape(M, H, K)
    v_proj = params.v_proj.reshape(M, H, V)
    o_proj = params.o_proj.reshape(H, R, V, M).transpose(1, 0, 2, 3)

    q = einsum(x, q_proj, 'B S M, M R H K -> B R H S K')
    k = einsum(x, k_proj, 'B D M, M   H K -> B   H D K')
    v = einsum(x, v_proj, 'B D M, M   H V -> B   H D V')

    q = apply_rotary_embedding(q, inputs.rope_cos, inputs.rope_sin)
    k = apply_rotary_embedding(k, inputs.rope_cos, inputs.rope_sin)

    qk_mask = inputs.attention_mask.astype(bool)
    qk_mask = jnp.tril(einsum(qk_mask, qk_mask, 'B i, B j -> B i j'))
    if config.sliding_window is not None:
        qk_mask = jnp.logical_and(qk_mask, jnp.triu(jnp.tril(jnp.full(
            qk_mask.shape, dtype=qk_mask.dtype, fill_value=True
        ), k=0), k=-config.sliding_window))
    qk_mask = qk_mask[:, None, None]

    qk = einsum(q, k, 'B R H S K, B H D K -> B R H S D') / math.sqrt(K)
    qk = jax.nn.softmax(qk, where=qk_mask, initial=0.)

    qkv = einsum(qk, v, 'B R H S D, B H D V -> B R H S V')
    out = einsum(qkv, o_proj, 'B R H S V, R H V M -> B S M')

    return out
