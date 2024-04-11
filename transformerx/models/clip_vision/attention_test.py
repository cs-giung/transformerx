# pylint: disable=duplicate-code
"""Testing Attention module."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import math
import numpy as np
import jax
import torch

from transformers import AutoConfig
from transformers.models.clip.modeling_clip import CLIPAttention
from transformerx.models.clip_vision.attention import \
    AttentionConfig, AttentionInputs, AttentionParams, forward_fn


if __name__ == '__main__':

    CONFIG = AutoConfig.from_pretrained('openai/clip-vit-large-patch14')

    BATCH_SIZE = 4
    SEQ_LEN = CONFIG.vision_config.image_size \
        // CONFIG.vision_config.patch_size # 224 // 14
    HIDDEN_SIZE = CONFIG.vision_config.hidden_size # 1024
    INTERMEDIATE_SIZE = CONFIG.vision_config.intermediate_size # 4096
    NUM_ATTENTION_HEADS = CONFIG.vision_config.num_attention_heads # 32

    # inputs
    inputs_jx = jax.random.normal(
        jax.random.PRNGKey(42), (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))

    # params
    q_proj_w_jx = math.sqrt(1.0 / HIDDEN_SIZE) * jax.random.normal(
        jax.random.PRNGKey(43), (HIDDEN_SIZE, HIDDEN_SIZE))
    k_proj_w_jx = math.sqrt(1.0 / HIDDEN_SIZE) * jax.random.normal(
        jax.random.PRNGKey(44), (HIDDEN_SIZE, HIDDEN_SIZE))
    v_proj_w_jx = math.sqrt(1.0 / HIDDEN_SIZE) * jax.random.normal(
        jax.random.PRNGKey(45), (HIDDEN_SIZE, HIDDEN_SIZE))
    o_proj_w_jx = math.sqrt(1.0 / HIDDEN_SIZE) * jax.random.normal(
        jax.random.PRNGKey(46), (HIDDEN_SIZE, HIDDEN_SIZE))

    q_proj_b_jx = jax.random.normal(jax.random.PRNGKey(47), (HIDDEN_SIZE,))
    k_proj_b_jx = jax.random.normal(jax.random.PRNGKey(48), (HIDDEN_SIZE,))
    v_proj_b_jx = jax.random.normal(jax.random.PRNGKey(49), (HIDDEN_SIZE,))
    o_proj_b_jx = jax.random.normal(jax.random.PRNGKey(50), (HIDDEN_SIZE,))

    # converters
    # pylint: disable=unnecessary-lambda-assignment
    jx2pt = lambda e: torch.from_numpy(np.asarray(e).copy())
    pt2np = lambda e: e.cpu().numpy()
    jx2np = lambda e: np.asarray(e).copy()

    # attention
    with torch.no_grad():
        inputs_pt = jx2pt(inputs_jx)
        module_pt = CLIPAttention(config=CONFIG.vision_config)
        module_pt.q_proj.weight.data.copy_(jx2pt(q_proj_w_jx.T))
        module_pt.k_proj.weight.data.copy_(jx2pt(k_proj_w_jx.T))
        module_pt.v_proj.weight.data.copy_(jx2pt(v_proj_w_jx.T))
        module_pt.out_proj.weight.data.copy_(jx2pt(o_proj_w_jx.T))
        module_pt.q_proj.bias.data.copy_(jx2pt(q_proj_b_jx))
        module_pt.k_proj.bias.data.copy_(jx2pt(k_proj_b_jx))
        module_pt.v_proj.bias.data.copy_(jx2pt(v_proj_b_jx))
        module_pt.out_proj.bias.data.copy_(jx2pt(o_proj_b_jx))
        output_pt = module_pt(inputs_pt)[0]

    for device in [jax.devices('cpu')[0]]:
        with jax.default_device(device):
            params = AttentionParams(
                q_proj_w = q_proj_w_jx,
                k_proj_w = k_proj_w_jx,
                v_proj_w = v_proj_w_jx,
                o_proj_w = o_proj_w_jx,
                q_proj_b = q_proj_b_jx,
                k_proj_b = k_proj_b_jx,
                v_proj_b = v_proj_b_jx,
                o_proj_b = o_proj_b_jx)
            inputs = AttentionInputs(hidden_states=inputs_jx)
            config = AttentionConfig(
                hidden_size=HIDDEN_SIZE,
                num_attention_heads=NUM_ATTENTION_HEADS)
            output = forward_fn(params, inputs, config)
            abserr = np.abs(jx2np(output) - pt2np(output_pt))
            print(f'torch - jax ({device})')
            print(f'- max: {abserr.max()}')
            print(f'- min: {abserr.min()}')
