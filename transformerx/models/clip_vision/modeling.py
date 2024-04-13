"""
CLIP Vision Model.

Classes:

Functions:
"""
from functools import partial
from typing import NamedTuple, Tuple

import jax.numpy as jnp
from einops import repeat

from transformerx.models.clip_vision.attention import \
    AttentionConfig, AttentionInputs, AttentionParams, \
    forward_fn as attention_fn
from transformerx.models.clip_vision.mlp import \
    MLPConfig, MLPInputs, MLPParams, \
    forward_fn as mlp_fn
from transformerx.models.clip_vision.normalization import \
    LayerNormConfig, LayerNormInputs, LayerNormParams, \
    forward_fn as layer_norm_fn
from transformerx.typing import Array, ArrayLike, PytreeLike


class CLIPVisionConfig(NamedTuple):
    """
    Attributes:
    """
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    patch_size: int
    projection_dim: int
    layer_norm_eps: float


class CLIPVisionInputs(NamedTuple): # pylint: disable=missing-class-docstring
    input_pixels: ArrayLike


class CLIPVisionOutput(NamedTuple): # pylint: disable=missing-class-docstring
    intermediates: Tuple[ArrayLike, ...]
    last_hidden_states: ArrayLike
    proj_hidden_states: ArrayLike


def embedding_fn(
        embedding_params: PytreeLike,
        input_pixels: ArrayLike,
        patch_size: int,
    ) -> Array:
    """Embedding function for the CLIP Vision Model."""

    # pylint: disable=invalid-name
    B, H, W, C = input_pixels.shape
    P = patch_size
    h = H // P
    w = W // P

    x = input_pixels
    x = x.transpose(0, 3, 1, 2) # [B, 3, 224, 224]
    x = x.reshape(B, C, h, P, w, P) # [B, 3, 16, 14, 16, 14]
    x = x.transpose(0, 2, 4, 1, 3, 5) # [B, 16, 16, 3, 14, 14]
    x = x.reshape(B, h * w, -1)

    x = x @ embedding_params['patch_embedding']['weight']
    x = jnp.concatenate((repeat(embedding_params[
        'class_embedding']['weight'], '1 i -> B 1 i', B=B), x), axis=1)
    x = x + embedding_params['position_embedding']['weight'][None]

    return x


@partial(jax.jit, static_argnames=('config', 'return_intermediates'))
def forward_fn(
        params: PytreeLike,
        inputs: CLIPVisionInputs,
        config: CLIPVisionConfig,
        **kwargs
    ) -> CLIPVisionOutput:
    """Forward function for the CLIP Vision Model."""

    # it collects intermediate hidden states if necessary
    intermediates = None
    if kwargs.pop('return_intermediates', False):
        intermediates = ()

    hidden_states = embedding_fn(
        params['embeddings'], inputs.input_pixels, config.patch_size)

    hidden_states = layer_norm_fn(
        params=LayerNormParams(
            weight=params['pre_layernorm']['weight'],
            bias=params['pre_layernorm']['bias']),
        inputs=LayerNormInputs(hidden_states=hidden_states),
        config=LayerNormConfig(layer_norm_eps=config.layer_norm_eps))

    for i in range(config.num_hidden_layers):

        residual = hidden_states
        hidden_states = layer_norm_fn(
            params=LayerNormParams(
                weight=params['layers'][f'{i}']['pre_layernorm']['weight'],
                bias=params['layers'][f'{i}']['pre_layernorm']['bias']),
            inputs=LayerNormInputs(hidden_states=hidden_states),
            config=LayerNormConfig(layer_norm_eps=config.layer_norm_eps))
        hidden_states = attention_fn(
            params=AttentionParams(
                q_proj_w=params[
                    'layers'][f'{i}']['self_attn']['q_proj']['weight'],
                k_proj_w=params[
                    'layers'][f'{i}']['self_attn']['k_proj']['weight'],
                v_proj_w=params[
                    'layers'][f'{i}']['self_attn']['v_proj']['weight'],
                o_proj_w=params[
                    'layers'][f'{i}']['self_attn']['o_proj']['weight'],
                q_proj_b=params[
                    'layers'][f'{i}']['self_attn']['q_proj']['bias'],
                k_proj_b=params[
                    'layers'][f'{i}']['self_attn']['k_proj']['bias'],
                v_proj_b=params[
                    'layers'][f'{i}']['self_attn']['v_proj']['bias'],
                o_proj_b=params[
                    'layers'][f'{i}']['self_attn']['o_proj']['bias']),
            inputs=AttentionInputs(hidden_states=hidden_states),
            config=AttentionConfig(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads))
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = layer_norm_fn(
            params=LayerNormParams(
                weight=params['layers'][f'{i}']['post_layernorm']['weight'],
                bias=params['layers'][f'{i}']['post_layernorm']['bias']),
            inputs=LayerNormInputs(hidden_states=hidden_states),
            config=LayerNormConfig(layer_norm_eps=config.layer_norm_eps))
        hidden_states = mlp_fn(
            params=MLPParams(
                u_proj_w=params['layers'][f'{i}']['mlp']['u_proj']['weight'],
                u_proj_b=params['layers'][f'{i}']['mlp']['u_proj']['bias'],
                d_proj_w=params['layers'][f'{i}']['mlp']['d_proj']['weight'],
                d_proj_b=params['layers'][f'{i}']['mlp']['d_proj']['bias']),
            inputs=MLPInputs(hidden_states=hidden_states),
            config=MLPConfig(intermediate_size=config.intermediate_size))
        hidden_states = hidden_states + residual

        if intermediates is not None:
            intermediates = intermediates + (hidden_states,)

    hidden_states = hidden_states[:, :1, :]
    hidden_states = layer_norm_fn(
        params=LayerNormParams(
            weight=params['post_layernorm']['weight'],
            bias=params['post_layernorm']['bias']),
        inputs=LayerNormInputs(hidden_states=hidden_states),
        config=LayerNormConfig(layer_norm_eps=config.layer_norm_eps))
    hidden_states = hidden_states[:, 0, :]

    proj_hidden_states = hidden_states @ params['projection']['weight']

    return CLIPVisionOutput(
        intermediates=intermediates,
        last_hidden_states=hidden_states,
        proj_hidden_states=proj_hidden_states)
