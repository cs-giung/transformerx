"""Utilities for making RoPE embeddings."""
import math
from typing import Tuple

import jax.numpy as jnp
from einops import einsum, repeat
from transformerx.typing import Array, ArrayLike


def make_simple_rope(
        position_ids: ArrayLike,
        dim: int,
        base: float,
    ) -> Tuple[Array, Array]:
    """Make rotary embedding based on position indices."""
    inv_freq = 1. / (base ** (jnp.arange(0, dim, 2) / dim))

    sinusoid = einsum(
        inv_freq, position_ids.astype(float), 'j, B L -> B L j')
    cos = repeat(jnp.cos(sinusoid), 'B L j -> B L (i j)', i=2)
    sin = repeat(jnp.sin(sinusoid), 'B L j -> B L (i j)', i=2)

    return cos, sin


def make_llama3_rope(
        position_ids: ArrayLike,
        dim: int,
        base: float,
        factor: float = 8.,
        low_freq_factor: float = 1.,
        high_freq_factor: float = 4.,
        original_max_position_embeddings: float = 8192,
    ) -> Tuple[Array, Array]:
    """Make rotary embedding based on position indices."""
    inv_freq = 1. / (base ** (jnp.arange(0, dim, 2) / dim))
    new_freq = []
    for freq in inv_freq:
        wavelen = 2. * math.pi / freq
        if wavelen < h_freq_wavelen:
            new_freq.append(freq)
            continue
        if wavelen > l_freq_wavelen:
            new_freq.append(freq / factor)
            continue
        assert l_freq_wavelen != h_freq_wavelen
        smooth = (old_context_len / wavelen - l_freq_factor)
        smooth = smooth / (h_freq_factor - l_freq_factor)
        new_freq.append((1. - smooth) * freq / factor + smooth * freq)
    inv_freq = jnp.array(new_freq)

    sinusoid = einsum(
        inv_freq, position_ids.astype(float), 'j, B L -> B L j')
    cos = repeat(jnp.cos(sinusoid), 'B L j -> B L (i j)', i=2)
    sin = repeat(jnp.sin(sinusoid), 'B L j -> B L (i j)', i=2)

    return cos, sin
