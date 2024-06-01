"""
Modeling Vision Transformer architecture.
https://arxiv.org/abs/2010.11929
"""
from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from einops import repeat

from transformerx.models.vit.attention import \
    AttentionConfig, AttentionInputs, AttentionParams, \
    forward_fn as attention_fn
from transformerx.models.vit.mlp import \
    MLPConfig, MLPInputs, MLPParams, \
    forward_fn as mlp_fn
from transformerx.models.vit.normalization import \
    LayerNormConfig, LayerNormInputs, LayerNormParams, \
    forward_fn as layer_norm_fn
from transformerx.typing import Array, ArrayLike, PytreeLike


class ViTConfig(NamedTuple):
    """
    Attributes:
        hidden_size (int): a dimension of the hidden representations.
        intermediate_size (int): an intermediate size in MLP modules.
        layer_norm_eps (float): an epsilon value for layer normalization.
        num_attention_heads (int): the number of attention heads.
        num_hidden_layers (int): the number of hidden layers.
        num_labels (int): the number of classes for classification.
        patch_size (int): a size of each patch.
        representation_size (int): a dimensionality of the pre-logit space.
    """
    hidden_size: int
    intermediate_size: int
    layer_norm_eps: float
    num_attention_heads: int
    num_hidden_layers: int
    num_labels: int
    patch_size: int
    representation_size: int


class ViTInputs(NamedTuple): # pylint: disable=missing-class-docstring
    input_pixels: ArrayLike


class ViTOutput(NamedTuple): # pylint: disable=missing-class-docstring
    intermediates: Tuple[ArrayLike, ...]
    last_hidden_states: ArrayLike
    pre_logits: ArrayLike
    logits: ArrayLike


def embedding_fn(
        embedding_params: PytreeLike,
        input_pixels: ArrayLike,
        patch_size: int,
    ) -> Array:
    """Embedding function for the Vision Transformer Model."""

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
    x = x + embedding_params['patch_embedding']['bias'][None, None, :]
    x = jnp.concatenate((repeat(embedding_params[
        'class_embedding']['weight'], '1 i -> B 1 i', B=B), x), axis=1)
    x = x + embedding_params['position_embedding']['weight'][None]

    return x


def block_fn(
        params: PytreeLike,
        inputs: ViTInputs, # pylint: disable=unused-argument
        config: ViTConfig,
        hidden: ArrayLike,
    ) -> Array:
    """Forward function for each transformer block."""
    residu = hidden
    hidden = layer_norm_fn(
        params=LayerNormParams(
            weight=params['pre_layernorm']['weight'],
            bias=params['pre_layernorm']['bias']),
        inputs=LayerNormInputs(hidden_states=hidden),
        config=LayerNormConfig(layer_norm_eps=config.layer_norm_eps))
    hidden = attention_fn(
        params=AttentionParams(
            q_proj_w=params['self_attn']['q_proj']['weight'],
            k_proj_w=params['self_attn']['k_proj']['weight'],
            v_proj_w=params['self_attn']['v_proj']['weight'],
            o_proj_w=params['self_attn']['o_proj']['weight'],
            q_proj_b=params['self_attn']['q_proj']['bias'],
            k_proj_b=params['self_attn']['k_proj']['bias'],
            v_proj_b=params['self_attn']['v_proj']['bias'],
            o_proj_b=params['self_attn']['o_proj']['bias']),
        inputs=AttentionInputs(hidden_states=hidden),
        config=AttentionConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads))
    hidden = hidden + residu

    residu = hidden
    hidden = layer_norm_fn(
        params=LayerNormParams(
            weight=params['post_layernorm']['weight'],
            bias=params['post_layernorm']['bias']),
        inputs=LayerNormInputs(hidden_states=hidden),
        config=LayerNormConfig(layer_norm_eps=config.layer_norm_eps))
    hidden = mlp_fn(
        params=MLPParams(
            u_proj_w=params['mlp']['u_proj']['weight'],
            u_proj_b=params['mlp']['u_proj']['bias'],
            d_proj_w=params['mlp']['d_proj']['weight'],
            d_proj_b=params['mlp']['d_proj']['bias']),
        inputs=MLPInputs(hidden_states=hidden),
        config=MLPConfig(intermediate_size=config.intermediate_size))
    hidden = hidden + residu

    return hidden


@partial(jax.jit, static_argnames=('config', 'return_intermediates'))
def forward_fn(
        params: PytreeLike,
        inputs: ViTInputs,
        config: ViTConfig,
        **kwargs
    ) -> ViTOutput:
    """Forward function for the Vision Transformer Model."""

    # it collects intermediate hidden states if necessary
    intermediates = None
    if kwargs.pop('return_intermediates', False):
        intermediates = ()

    hidden_states = embedding_fn(
        params['embeddings'], inputs.input_pixels, config.patch_size)

    for i in range(config.num_hidden_layers):
        hidden_states = block_fn(
            params['layers'][f'{i}'], inputs, config, hidden=hidden_states)
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

    pre_logits = None
    if 'pooler' in params:
        pre_logits = hidden_states @ params['pooler']['weight']
        pre_logits = pre_logits + params['pooler']['bias'][None, :]
        pre_logits = jax.nn.tanh(pre_logits)
        hidden_states = pre_logits

    logits = None
    if 'head' in params:
        logits = hidden_states @ params['head']['weight']
        logits = logits + params['head']['bias'][None, :]

    return ViTOutput(
        intermediates=intermediates,
        last_hidden_states=hidden_states,
        pre_logits=pre_logits,
        logits=logits)
