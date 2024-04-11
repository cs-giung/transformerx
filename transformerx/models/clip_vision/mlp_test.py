# pylint: disable=duplicate-code
"""Testing MLP block module."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import math
import numpy as np
import jax
import torch

from transformers import AutoConfig
from transformers.models.clip.modeling_clip import CLIPMLP
from transformerx.models.clip_vision.mlp import \
    MLPParams, MLPInputs, MLPConfig, forward_fn


if __name__ == '__main__':

    CONFIG = AutoConfig.from_pretrained('openai/clip-vit-large-patch14')

    BATCH_SIZE = 4
    SEQ_LEN = CONFIG.vision_config.image_size \
        // CONFIG.vision_config.patch_size # 224 // 14
    HIDDEN_SIZE = CONFIG.vision_config.hidden_size # 1024
    INTERMEDIATE_SIZE = CONFIG.vision_config.intermediate_size # 4096

    # inputs
    inputs_jx = jax.random.normal(
        jax.random.PRNGKey(42), (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))

    # params
    fc1_w_jx = math.sqrt(1.0 / HIDDEN_SIZE) * jax.random.normal(
        jax.random.PRNGKey(43), (HIDDEN_SIZE, INTERMEDIATE_SIZE))
    fc1_b_jx = jax.random.normal(jax.random.PRNGKey(44), (INTERMEDIATE_SIZE,))
    fc2_w_jx = math.sqrt(1.0 / INTERMEDIATE_SIZE) * jax.random.normal(
        jax.random.PRNGKey(43), (INTERMEDIATE_SIZE, HIDDEN_SIZE))
    fc2_b_jx = jax.random.normal(jax.random.PRNGKey(44), (HIDDEN_SIZE,))

    # converters
    # pylint: disable=unnecessary-lambda-assignment
    jx2pt = lambda e: torch.from_numpy(np.asarray(e).copy())
    pt2np = lambda e: e.cpu().numpy()
    jx2np = lambda e: np.asarray(e).copy()

    with torch.no_grad():
        inputs_pt = jx2pt(inputs_jx)
        fc1_w_pt = jx2pt(fc1_w_jx).T
        fc2_w_pt = jx2pt(fc2_w_jx).T
        fc1_b_pt = jx2pt(fc1_b_jx)
        fc2_b_pt = jx2pt(fc2_b_jx)
        module_pt = CLIPMLP(config=CONFIG.vision_config)
        module_pt.fc1.weight.data.copy_(fc1_w_pt)
        module_pt.fc2.weight.data.copy_(fc2_w_pt)
        module_pt.fc1.bias.data.copy_(fc1_b_pt)
        module_pt.fc2.bias.data.copy_(fc2_b_pt)
        output_pt = module_pt(inputs_pt)

    for device in [jax.devices('cpu')[0]]:
        with jax.default_device(device):
            params = MLPParams(
                u_proj_w=fc1_w_jx, u_proj_b=fc1_b_jx,
                d_proj_w=fc2_w_jx, d_proj_b=fc2_b_jx)
            inputs = MLPInputs(hidden_states=inputs_jx)
            config = MLPConfig(intermediate_size=INTERMEDIATE_SIZE)
            output = forward_fn(params, inputs, config)
            abserr = np.abs(jx2np(output) - pt2np(output_pt))
            print(f'torch - jax ({device})')
            print(f'- max: {abserr.max()}')
            print(f'- min: {abserr.min()}')
