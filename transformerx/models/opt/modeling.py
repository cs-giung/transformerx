"""
OPT: Open Pre-trained Transformer Language Models
https://arxiv.org/abs/2205.01068
"""
from functools import partial
from typing import NamedTuple, Tuple

import jax
from transformerx.models.opt.attention import \
    AttentionParams, AttentionInputs, AttentionConfig, \
    forward_fn as attention_fn
from transformerx.models.opt.mlp import \
    MLPParams, MLPInputs, MLPConfig, \
    forward_fn as mlp_fn
from transformerx.models.opt.normalization import \
    LayerNormParams, LayerNormInputs, LayerNormConfig, \
    forward_fn as layer_norm_fn
from transformerx.typing import Array, ArrayLike, PytreeLike


class OPTConfig(NamedTuple):
    """
    Attributes:
        hidden_size (int): a dimension of the hidden representations.
        intermediate_size (int): an intermediate size in MLP modules.
        layer_norm_eps (float): an epsilon vlue for layer normalization.
        num_attention_heads (int): the number of attention heads.
        num_hidden_layers (int): the number of hidden layers.
        vocab_size (int): vocabulary size of the text model.
    """
    hidden_size: int
    intermediate_size: int
    layer_norm_eps: float
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int


class OPTInputs(NamedTuple): # pylint: disable=missing-class-docstring
    input_ids: ArrayLike
    attention_mask: ArrayLike
    position_ids: ArrayLike


class OPTOutput(NamedTuple): # pylint: disable=missing-class-docstring
    intermediates: Tuple[ArrayLike, ...]
    last_hidden_states: ArrayLike
    logits: ArrayLike


def block_fn(
        params: PytreeLike,
        inputs: OPTInputs,
        config: OPTConfig,
        hidden: ArrayLike,
    ) -> Array:
    """Forward function for each Transformer block."""
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
            q_proj_b=params['self_attn']['q_proj']['bias'],
            k_proj_w=params['self_attn']['k_proj']['weight'],
            k_proj_b=params['self_attn']['k_proj']['bias'],
            v_proj_w=params['self_attn']['v_proj']['weight'],
            v_proj_b=params['self_attn']['v_proj']['bias'],
            o_proj_w=params['self_attn']['o_proj']['weight'],
            o_proj_b=params['self_attn']['o_proj']['bias']),
        inputs=AttentionInputs(
            hidden_states=hidden,
            attention_mask=inputs.attention_mask),
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
        inputs: OPTInputs,
        config: OPTConfig,
        **kwargs
    ) -> OPTOutput:
    """Forward function for the OPT model."""

    # it collects intermediate hidden states if necessary
    intermediates = None
    if kwargs.pop('return_intermediates', False):
        intermediates = ()

    pos_embed = (
        jax.numpy.cumsum(inputs.attention_mask, axis=1) * inputs.attention_mask
    ).astype(jax.numpy.int32) - 1
    hidden_states = params[
        'embeddings']['token_embedding']['weight'][inputs.input_ids] + params[
        'embeddings']['position_embedding']['weight'][pos_embed + 2]

    for i in range(config.num_hidden_layers):
        hidden_states = block_fn(
            params['layers'][f'{i}'], inputs, config, hidden=hidden_states)
        if intermediates is not None:
            intermediates = intermediates + (hidden_states,)

    hidden_states = layer_norm_fn(
        params=LayerNormParams(
            weight=params['final_layer_norm']['weight'],
            bias=params['final_layer_norm']['bias']),
        inputs=LayerNormInputs(hidden_states=hidden_states),
        config=LayerNormConfig(layer_norm_eps=config.layer_norm_eps))

    logits = None
    if 'lm_head' in params:
        logits = hidden_states @ params['lm_head']['weight']

    return OPTOutput(
        intermediates=intermediates,
        last_hidden_states=hidden_states,
        logits=logits)
