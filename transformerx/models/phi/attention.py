"""Functions and classes for the attention module."""
import math
from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from einops import einsum, rearrange, repeat
from transformerx.typing import Array, ArrayLike


class AttentionParams(NamedTuple): # pylint: disable=missing-class-docstring
    q_proj: ArrayLike
    k_proj: ArrayLike
    v_proj: ArrayLike
    o_proj: ArrayLike


class AttentionInputs(NamedTuple): # pylint: disable=missing-class-docstring
    hidden_states: ArrayLike
    attention_mask: ArrayLike
    position_ids: ArrayLike


class AttentionConfig(NamedTuple): # pylint: disable=missing-class-docstring
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rope_theta: float


def make_rotary_embedding(
        position_ids: ArrayLike,
        d_k: int,
        rope_theta: float,
    ) -> Tuple[Array, Array]:
    """Make rotary embedding based on position indices."""
    inv_freq = 1. / (rope_theta ** (jnp.arange(0, d_k, 2) / d_k))
    sinusoid = einsum(inv_freq, position_ids.astype(float), 'j, B L -> B L j')
    cos = repeat(jnp.cos(sinusoid), 'B L j -> B L (i j)', i=2)
    sin = repeat(jnp.sin(sinusoid), 'B L j -> B L (i j)', i=2)
    return cos, sin


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

    cos, sin = make_rotary_embedding(inputs.position_ids, K, config.rope_theta)
    q = apply_rotary_embedding(q, cos, sin)
    k = apply_rotary_embedding(k, cos, sin)

    qk_mask = inputs.attention_mask.astype(bool)
    qk_mask = jnp.tril(einsum(qk_mask, qk_mask, 'B i, B j -> B i j'))
    qk_mask = qk_mask[:, None, None]

    qk = einsum(q, k, 'B R H S K, B H D K -> B R H S D') / math.sqrt(K)
    qk = jax.nn.softmax(qk, where=qk_mask, initial=0.)

    qkv = einsum(qk, v, 'B R H S D, B H D V -> B R H S V')
    out = einsum(qkv, o_proj, 'B R H S V, R H V M -> B S M')

    return out
