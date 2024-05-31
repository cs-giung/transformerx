"""
Modeling Text Transformer architecture in CLIP.
https://arxiv.org/abs/2103.00020
"""
from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from transformerx.models.clip_txt.attention import \
    AttentionConfig, AttentionInputs, AttentionParams, \
    forward_fn as attention_fn
from transformerx.models.clip_txt.mlp import \
    MLPConfig, MLPInputs, MLPParams, \
    forward_fn as mlp_fn
from transformerx.models.clip_txt.normalization import \
    LayerNormConfig, LayerNormInputs, LayerNormParams, \
    forward_fn as layer_norm_fn
from transformerx.typing import ArrayLike, PytreeLike


class CLIPTxTConfig(NamedTuple):
    """
    Attributes:
        hidden_act (str): an activation function in MLP modules.
        hidden_size (int): a dimension of the hidden representations.
        intermediate_size (int): an intermediate size in MLP modules.
        num_attention_heads (int): the number of attention heads.
        num_hidden_layers (int): the number of hidden layers.
        projection_dim (int): a dimensionality of the shared embedding space.
        layer_norm_eps (float): an epsilon value for layer normalization.
        vocab_size (int):
    """
    hidden_act: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    projection_dim: int
    layer_norm_eps: float
    vocab_size: int


class CLIPTxTInputs(NamedTuple): # pylint: disable=missing-class-docstring
    input_ids: ArrayLike
    attention_mask: ArrayLike
    position_ids: ArrayLike


class CLIPTxTOutput(NamedTuple): # pylint: disable=missing-class-docstring
    intermediates: Tuple[ArrayLike, ...]
    last_hidden_states: ArrayLike
    proj_hidden_states: ArrayLike


@partial(jax.jit, static_argnames=('config', 'return_intermediates'))
def forward_fn(
        params: PytreeLike,
        inputs: CLIPTxTInputs,
        config: CLIPTxTConfig,
        **kwargs
    ) -> CLIPTxTOutput:
    """Forward function for the CLIP Text Model."""

    # it collects intermediate hidden states if necessary
    intermediates = None
    if kwargs.pop('return_intermediates', False):
        intermediates = ()

    hidden_states = params[
        'embeddings']['token_embedding']['weight'][inputs.input_ids] + params[
        'embeddings']['position_embedding']['weight'][inputs.position_ids]

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
            inputs=AttentionInputs(
                hidden_states=hidden_states,
                attention_mask=inputs.attention_mask),
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
            config=MLPConfig(
                hidden_act=config.hidden_act,
                intermediate_size=config.intermediate_size))
        hidden_states = hidden_states + residual

        if intermediates is not None:
            intermediates = intermediates + (hidden_states,)

    hidden_states = jnp.take_along_axis(
        hidden_states, jnp.argmax(inputs.position_ids, axis=1)[:, None, None], axis=1)
    hidden_states = layer_norm_fn(
        params=LayerNormParams(
            weight=params['final_layer_norm']['weight'],
            bias=params['final_layer_norm']['bias']),
        inputs=LayerNormInputs(hidden_states=hidden_states),
        config=LayerNormConfig(layer_norm_eps=config.layer_norm_eps))
    hidden_states = hidden_states[:, 0, :]

    proj_hidden_states = hidden_states @ params['projection']['weight']

    return CLIPTxTOutput(
        intermediates=intermediates,
        last_hidden_states=hidden_states,
        proj_hidden_states=proj_hidden_states)
