# pylint: disable=duplicate-code
"""Testing layer normalization module."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import math
import numpy as np
import jax
import torch

from torch.nn import LayerNorm
from transformerx.models.clip_vision.normalization import \
    LayerNormParams, LayerNormInputs, LayerNormConfig, forward_fn


if __name__ == '__main__':

    BATCH_SIZE = 4
    HIDDEN_SIZE = 768
    LAYER_NORM_EPS = 1e-05

    # inputs
    inputs_jx = jax.random.normal(
        jax.random.PRNGKey(42), (BATCH_SIZE, HIDDEN_SIZE))

    # params
    weight_jx = jax.random.normal(jax.random.PRNGKey(43), (HIDDEN_SIZE,))
    bias_jx = jax.random.normal(jax.random.PRNGKey(44), (HIDDEN_SIZE,))

    # converters
    # pylint: disable=unnecessary-lambda-assignment
    jx2pt = lambda e: torch.from_numpy(np.asarray(e).copy())
    pt2np = lambda e: e.cpu().numpy()
    jx2np = lambda e: np.asarray(e).copy()

    with torch.no_grad():
        inputs_pt = jx2pt(inputs_jx)
        weight_pt = jx2pt(weight_jx)
        bias_pt = jx2pt(bias_jx)
        module_pt = LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)
        module_pt.weight.data.copy_(weight_pt)
        module_pt.bias.data.copy_(bias_pt)
        output_pt = module_pt(inputs_pt)

    for device in [jax.devices('cpu')[0]]:
        with jax.default_device(device):
            params = LayerNormParams(weight=weight_jx, bias=bias_jx)
            inputs = LayerNormInputs(hidden_states=inputs_jx)
            config = LayerNormConfig(layer_norm_eps=LAYER_NORM_EPS)
            output = forward_fn(params, inputs, config)
            abserr = np.abs(jx2np(output) - pt2np(output_pt))
            print(f'torch - jax ({device})')
            print(f'- max: {abserr.max()}')
            print(f'- min: {abserr.min()}')
