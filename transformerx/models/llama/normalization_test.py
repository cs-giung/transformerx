# pylint: disable=duplicate-code
"""Testing RMS Normalization Module."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import math
import numpy as np
import jax
import torch

from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformerx.models.llama.normalization import \
    RMSNormParams, RMSNormInputs, RMSNormConfig, forward_fn


if __name__ == '__main__':

    CONFIG = AutoConfig.from_pretrained('meta-llama/llama-2-7b-hf')

    BATCH_SIZE = 4
    SEQ_LEN = CONFIG.max_position_embeddings # 4096
    HIDDEN_SIZE = CONFIG.hidden_size # 4096
    RMS_NORM_EPS = CONFIG.rms_norm_eps # 1e-05

    # inputs
    inputs_jx = jax.random.normal(
        jax.random.PRNGKey(42), (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))

    # params
    weight_jx = math.sqrt(1.0 / HIDDEN_SIZE) * jax.random.normal(
        jax.random.PRNGKey(43), (HIDDEN_SIZE,))

    # converters
    # pylint: disable=unnecessary-lambda-assignment
    jx2pt = lambda e: torch.from_numpy(np.asarray(e).copy())
    pt2np = lambda e: e.cpu().numpy()
    jx2np = lambda e: np.asarray(e).copy()

    with torch.no_grad():
        inputs_pt = jx2pt(inputs_jx)
        weight_pt = jx2pt(weight_jx)
        module_pt = LlamaRMSNorm(hidden_size=HIDDEN_SIZE, eps=RMS_NORM_EPS)
        module_pt.weight.data.copy_(weight_pt)
        output_pt = module_pt(inputs_pt)

    for device in [jax.devices('cpu')[0], jax.devices()[0]]:
        with jax.default_device(device):
            params = RMSNormParams(weight=weight_jx)
            inputs = RMSNormInputs(hidden_states=inputs_jx)
            config = RMSNormConfig(rms_norm_eps=RMS_NORM_EPS)
            output = forward_fn(params, inputs, config)
            abserr = np.abs(jx2np(output) - pt2np(output_pt))
            print(f'torch - jax ({device})')
            print(f'- max: {abserr.max()}')
            print(f'- min: {abserr.min()}')
