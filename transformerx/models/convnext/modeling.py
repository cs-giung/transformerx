"""
Modeling ConvNext architecture.
https://arxiv.org/abs/2201.03545
"""
from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from transformerx.models.convnext.normalization import \
    LayerNormConfig, LayerNormInputs, LayerNormParams, \
    forward_fn as layer_norm_fn
from transformerx.typing import Array, ArrayLike, PytreeLike


class ConvNextConfig(NamedTuple):
    """
    Attributes:
    """
    hidden_sizes: Tuple[int]
    layer_norm_eps: float
    num_hidden_layers: Tuple[int]
    num_labels: int
    patch_size: int


class ConvNextInputs(NamedTuple): # pylint: disable=missing-class-docstring
    input_pixels: ArrayLike


class ConvNextOutput(NamedTuple): # pylint: disable=missing-class-docstring
    intermediates: Tuple[ArrayLike, ...]
    last_hidden_states: ArrayLike
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

    return x.reshape(B, h, w, -1)


def block_fn(
        params: PytreeLike,
        inputs: ConvNextInputs, # pylint: disable=unused-argument
        config: ConvNextConfig,
        hidden: ArrayLike
    ) -> Array:
    """Forward function for each block."""
    if hidden.shape[-1] != params['dwconv']['weight'].shape[-1]:
        hidden = layer_norm_fn(
            params=LayerNormParams(
                weight=params['dsnorm']['weight'],
                bias=params['dsnorm']['bias']),
            inputs=LayerNormInputs(hidden_states=hidden),
            config=LayerNormConfig(layer_norm_eps=config.layer_norm_eps))
        hidden = jax.lax.conv_general_dilated(
            hidden, params['dsconv']['weight'],
            window_strides=params['dsconv']['weight'].shape[:2], padding='SAME',
            dimension_numbers=jax.lax.ConvDimensionNumbers(
                (0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2)))
        hidden = hidden + params['dsconv']['bias'][None, None, None]

    residu = hidden

    hidden = jax.lax.conv_general_dilated(
        hidden, params['dwconv']['weight'],
        window_strides=(1, 1), padding='SAME',
        dimension_numbers=jax.lax.ConvDimensionNumbers(
            (0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2)),
        feature_group_count=hidden.shape[-1])
    hidden = hidden + params['dwconv']['bias'][None, None, None]
    hidden = layer_norm_fn(
        params=LayerNormParams(
            weight=params['layernorm']['weight'],
            bias=params['layernorm']['bias']),
        inputs=LayerNormInputs(hidden_states=hidden),
        config=LayerNormConfig(layer_norm_eps=config.layer_norm_eps))

    hidden = hidden @ params['pwconv1']['weight']
    hidden = hidden + params['pwconv1']['bias'][None, None, None]
    hidden = jax.nn.gelu(hidden, approximate=False)

    hidden = hidden @ params['pwconv2']['weight']
    hidden = hidden + params['pwconv2']['bias'][None, None, None]
    hidden = hidden * params['layerscale']['weight']

    hidden = hidden + residu

    return hidden


@partial(jax.jit, static_argnames=('config', 'return_intermediates'))
def forward_fn(
        params: PytreeLike,
        inputs: ConvNextInputs,
        config: ConvNextConfig,
        **kwargs
    ) -> ConvNextOutput:
    """Forward function for the ConvNext V2 Model."""

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

    for i, num_hidden_layers in enumerate(config.num_hidden_layers):
        for j in range(num_hidden_layers):
            hidden_states = block_fn(
                params['layers'][f'{i}'][f'{j}'], inputs, config, hidden_states)
        if intermediates is not None:
            intermediates = intermediates + (hidden_states,)

    hidden_states = jnp.mean(hidden_states, axis=(-3, -2), keepdims=True)
    hidden_states = layer_norm_fn(
        params=LayerNormParams(
            weight=params['post_layernorm']['weight'],
            bias=params['post_layernorm']['bias']),
        inputs=LayerNormInputs(hidden_states=hidden_states),
        config=LayerNormConfig(layer_norm_eps=config.layer_norm_eps))
    hidden_states = hidden_states[:, 0, 0, :]

    logits = None
    if 'head' in params:
        logits = hidden_states @ params['head']['weight']
        logits = logits + params['head']['bias'][None, :]

    return ConvNextOutput(
        intermediates=intermediates,
        last_hidden_states=hidden_states,
        logits=logits)
