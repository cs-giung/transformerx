"""
Llama Model.

Classes:
    LlamaParams:
    LlamaInputs:
    LlamaConfig:

Functions:
    forward_fn: Forward function for the Llama model.
"""
from functools import partial
from typing import NamedTuple, Tuple

import jax
from transformerx.models.llama.attention import \
    AttentionParams, AttentionInputs, AttentionConfig, \
    forward_fn as attention_fn
from transformerx.models.llama.mlp import \
    MLPParams, MLPInputs, MLPConfig, \
    forward_fn as mlp_fn
from transformerx.models.llama.normalization import \
    RMSNormParams, RMSNormInputs, RMSNormConfig, \
    forward_fn as rms_norm_fn
from transformerx.typing import ArrayLike, PytreeLike


class LlamaConfig(NamedTuple):
    """
    Attributes:
        hidden_size (int): a dimension of the hidden representations.
        intermediate_size (int): an intermediate size in MLP modules.
        num_attention_heads (int): the number of attention heads.
        num_hidden_layers (int): the number of hidden layers.
        num_key_value_heads (int): the number of heads for keys and values in
            grouped-query attention. When it equals to `num_attention_heads`,
            multi-head attention is used. If it is set to one, multi-query
            attention is applied.
        rms_norm_eps (float): an epsilon value for RMS normalization.
    """
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int


class LlamaInputs(NamedTuple): # pylint: disable=missing-class-docstring
    input_ids: ArrayLike
    attention_mask: ArrayLike
    position_ids: ArrayLike


class LlamaOutput(NamedTuple): # pylint: disable=missing-class-docstring
    intermediates: Tuple[ArrayLike, ...]
    last_hidden_states: ArrayLike
    logits: ArrayLike


@partial(jax.jit, static_argnames=('config', 'return_intermediates'))
def forward_fn(
        params: PytreeLike,
        inputs: LlamaInputs,
        config: LlamaConfig,
        **kwargs
    ) -> LlamaOutput:
    """Forward function for the Llama model."""

    # it collects intermediate hidden states if necessary
    intermediates = None
    if kwargs.pop('return_intermediates', False):
        intermediates = ()

    hidden_states = params['embed_tokens']['weight'][inputs.input_ids]

    for i in range(config.num_hidden_layers):

        residual = hidden_states
        hidden_states = rms_norm_fn(
            params=RMSNormParams(
                weight=params['layers'][f'{i}']['input_layernorm']['weight']),
            inputs=RMSNormInputs(hidden_states=hidden_states),
            config=RMSNormConfig(rms_norm_eps=config.rms_norm_eps))
        hidden_states = attention_fn(
            params=AttentionParams(
                q_proj=params['layers'][f'{i}']['self_attn']['q_proj']['weight'],
                k_proj=params['layers'][f'{i}']['self_attn']['k_proj']['weight'],
                v_proj=params['layers'][f'{i}']['self_attn']['v_proj']['weight'],
                o_proj=params['layers'][f'{i}']['self_attn']['o_proj']['weight']),
            inputs=AttentionInputs(
                hidden_states=hidden_states,
                attention_mask=inputs.attention_mask,
                position_ids=inputs.position_ids),
            config=AttentionConfig(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads))
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = rms_norm_fn(
            params=RMSNormParams(
                weight=params['layers'][f'{i}']['post_attention_layernorm']['weight']),
            inputs=RMSNormInputs(hidden_states=hidden_states),
            config=RMSNormConfig(rms_norm_eps=config.rms_norm_eps))
        hidden_states = mlp_fn(
            params=MLPParams(
                g_proj=params['layers'][f'{i}']['mlp']['g_proj']['weight'],
                u_proj=params['layers'][f'{i}']['mlp']['u_proj']['weight'],
                d_proj=params['layers'][f'{i}']['mlp']['d_proj']['weight']),
            inputs=MLPInputs(hidden_states=hidden_states),
            config=MLPConfig(intermediate_size=config.intermediate_size))
        hidden_states = hidden_states + residual

        if intermediates is not None:
            intermediates = intermediates + (hidden_states,)

    hidden_states = rms_norm_fn(
        params=RMSNormParams(
            weight=params['norm']['weight']),
        inputs=RMSNormInputs(hidden_states=hidden_states),
        config=RMSNormConfig(rms_norm_eps=config.rms_norm_eps))

    logits = hidden_states @ params['lm_head']['weight']

    return LlamaOutput(
        intermediates=intermediates,
        last_hidden_states=hidden_states, logits=logits)
