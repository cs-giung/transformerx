"""Utilities for making RoPE embeddings."""
from typing import Tuple

import jax.numpy as jnp
from einops import einsum, repeat
from transformerx.typing import Array, ArrayLike


def make_rope(
        position_ids: ArrayLike,
        d_k: int,
        rope_kwargs: dict,
    ) -> Tuple[Array, Array]:
    """Make rotary embedding based on position indices."""
    base = rope_kwargs['base']
    inv_freq = 1. / (base ** (jnp.arange(0, d_k, 2) / d_k))
    sinusoid = einsum(
        inv_freq, position_ids.astype(float), 'j, B L -> B L j')
    cos = repeat(jnp.cos(sinusoid), 'B L j -> B L (i j)', i=2)
    sin = repeat(jnp.sin(sinusoid), 'B L j -> B L (i j)', i=2)
    return cos, sin
