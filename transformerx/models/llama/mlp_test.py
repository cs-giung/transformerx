# pylint: disable=duplicate-code
"""Testing MLP Block Module."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import math
import numpy as np
import jax
import torch

from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from transformerx.models.llama.mlp import \
    MLPParams, MLPInputs, MLPConfig, forward_fn


if __name__ == '__main__':

    CONFIG = AutoConfig.from_pretrained('meta-llama/llama-2-7b-hf')

    BATCH_SIZE = 4
    SEQ_LEN = CONFIG.max_position_embeddings # 4096
    HIDDEN_SIZE = CONFIG.hidden_size # 4096
    INTERMEDIATE_SIZE = CONFIG.intermediate_size # 11008

    # inputs
    inputs_jx = jax.random.normal(
        jax.random.PRNGKey(42), (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))

    # params
    g_proj_jx = math.sqrt(1.0 / HIDDEN_SIZE) * jax.random.normal(
        jax.random.PRNGKey(43), (HIDDEN_SIZE, INTERMEDIATE_SIZE))
    u_proj_jx = math.sqrt(1.0 / HIDDEN_SIZE) * jax.random.normal(
        jax.random.PRNGKey(44), (HIDDEN_SIZE, INTERMEDIATE_SIZE))
    d_proj_jx = math.sqrt(1.0 / INTERMEDIATE_SIZE) * jax.random.normal(
        jax.random.PRNGKey(45), (INTERMEDIATE_SIZE, HIDDEN_SIZE))

    # converters
    # pylint: disable=unnecessary-lambda-assignment
    jx2pt = lambda e: torch.from_numpy(np.asarray(e).copy())
    pt2np = lambda e: e.cpu().numpy()
    jx2np = lambda e: np.asarray(e).copy()

    with torch.no_grad():
        inputs_pt = jx2pt(inputs_jx)
        g_proj_pt = jx2pt(g_proj_jx).T
        u_proj_pt = jx2pt(u_proj_jx).T
        d_proj_pt = jx2pt(d_proj_jx).T
        module_pt = LlamaMLP(config=CONFIG)
        module_pt.gate_proj.weight.data.copy_(g_proj_pt)
        module_pt.up_proj.weight.data.copy_(u_proj_pt)
        module_pt.down_proj.weight.data.copy_(d_proj_pt)
        output_pt = module_pt(inputs_pt)

    for device in [jax.devices('cpu')[0], jax.devices()[0]]:
        with jax.default_device(device):
            params = MLPParams(
                g_proj=g_proj_jx, u_proj=u_proj_jx, d_proj=d_proj_jx)
            inputs = MLPInputs(hidden_states=inputs_jx)
            config = MLPConfig(intermediate_size=INTERMEDIATE_SIZE)
            output = forward_fn(params, inputs, config)
            abserr = np.abs(jx2np(output) - pt2np(output_pt))
            print(f'torch - jax ({device})')
            print(f'- max: {abserr.max()}')
            print(f'- min: {abserr.min()}')
