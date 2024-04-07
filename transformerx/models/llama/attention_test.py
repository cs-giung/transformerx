# pylint: disable=duplicate-code
"""Testing Attention Module."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import math
import numpy as np
import jax
import torch

from transformers import AutoConfig
from transformers.models.llama.modeling_llama import \
    LlamaRotaryEmbedding, LlamaAttention
from transformerx.models.llama.attention import \
    AttentionParams, AttentionInputs, AttentionConfig, \
    make_rotary_embedding, forward_fn


if __name__ == '__main__':

    CONFIG = AutoConfig.from_pretrained('meta-llama/llama-2-7b-hf')

    BATCH_SIZE = 4
    SEQ_LEN = CONFIG.max_position_embeddings # 4096
    HIDDEN_SIZE = CONFIG.hidden_size # 4096
    MAX_POSITION_EMBEDDINGS = CONFIG.max_position_embeddings
    NUM_ATTENTION_HEADS = CONFIG.num_attention_heads # 32
    NUM_KEY_VALUE_HEADS = CONFIG.num_key_value_heads # 32

    # inputs
    inputs_jx = jax.random.normal(
        jax.random.PRNGKey(42), (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))
    position_ids_jx = jax.numpy.array([
        jax.numpy.arange(SEQ_LEN) for _ in range(BATCH_SIZE)]).astype(int)
    attention_mask_jx = jax.numpy.array([
        jax.numpy.concatenate([
            jax.numpy.arange(i * 100),
            jax.numpy.zeros(SEQ_LEN - i * 100)])
        for i in range(BATCH_SIZE)]).astype(int)

    # params
    q_proj_jx = math.sqrt(1.0 / HIDDEN_SIZE) * jax.random.normal(
        jax.random.PRNGKey(43), (HIDDEN_SIZE, HIDDEN_SIZE))
    k_proj_jx = math.sqrt(1.0 / HIDDEN_SIZE) * jax.random.normal(
        jax.random.PRNGKey(44), (
            HIDDEN_SIZE,
            HIDDEN_SIZE // NUM_ATTENTION_HEADS * NUM_KEY_VALUE_HEADS))
    v_proj_jx = math.sqrt(1.0 / HIDDEN_SIZE) * jax.random.normal(
        jax.random.PRNGKey(45), (
            HIDDEN_SIZE,
            HIDDEN_SIZE // NUM_ATTENTION_HEADS * NUM_KEY_VALUE_HEADS))
    o_proj_jx = math.sqrt(1.0 / HIDDEN_SIZE) * jax.random.normal(
        jax.random.PRNGKey(46), (HIDDEN_SIZE, HIDDEN_SIZE))

    # converters
    # pylint: disable=unnecessary-lambda-assignment
    jx2pt = lambda e: torch.from_numpy(np.asarray(e).copy())
    pt2np = lambda e: e.cpu().numpy()
    jx2np = lambda e: np.asarray(e).copy()

    # rotary embedding
    with torch.no_grad():
        inputs_pt = jx2pt(inputs_jx)
        position_ids_pt = jx2pt(position_ids_jx)
        module_pt = LlamaRotaryEmbedding(
            dim=(HIDDEN_SIZE // NUM_ATTENTION_HEADS),
            max_position_embeddings=MAX_POSITION_EMBEDDINGS)
        output_pt = module_pt(inputs_pt, position_ids_pt)

    for device in [jax.devices('cpu')[0], jax.devices()[0]]:
        with jax.default_device(device):
            output = make_rotary_embedding(
                position_ids_jx, (HIDDEN_SIZE // NUM_ATTENTION_HEADS))
            print(f'torch - jax ({device})')
            abserr = np.abs(jx2np(output[0][0]) - pt2np(output_pt[0]))
            print(f'- max: {abserr.max()}')
            print(f'- min: {abserr.min()}')
            abserr = np.abs(jx2np(output[1][0]) - pt2np(output_pt[1]))
            print(f'- max: {abserr.max()}')
            print(f'- min: {abserr.min()}')

    # attention
    with torch.no_grad():
        inputs_pt = jx2pt(inputs_jx)
        position_ids_pt = jx2pt(position_ids_jx)
        attention_mask_pt = jx2pt(attention_mask_jx)
        attention_mask_pt = attention_mask_pt.bool()
        attention_mask_pt = torch.tril(
            torch.einsum('bi,bj->bij', attention_mask_pt, attention_mask_pt)
            )[:, None]
        attention_mask_pt = torch.where(
            attention_mask_pt, 0, torch.finfo(torch.float32).min)
        module_pt = LlamaAttention(config=CONFIG, layer_idx=0)
        module_pt.q_proj.weight.data.copy_(jx2pt(q_proj_jx.T))
        module_pt.k_proj.weight.data.copy_(jx2pt(k_proj_jx.T))
        module_pt.v_proj.weight.data.copy_(jx2pt(v_proj_jx.T))
        module_pt.o_proj.weight.data.copy_(jx2pt(o_proj_jx.T))
        output_pt = module_pt(
            inputs_pt,
            attention_mask=attention_mask_pt,
            position_ids=position_ids_pt)[0]

    for device in [jax.devices('cpu')[0], jax.devices()[0]]:
        with jax.default_device(device):
            params = AttentionParams(
                q_proj=q_proj_jx,
                k_proj=k_proj_jx,
                v_proj=v_proj_jx,
                o_proj=o_proj_jx)
            inputs = AttentionInputs(
                hidden_states=inputs_jx,
                attention_mask=attention_mask_jx,
                position_ids=position_ids_jx)
            config  = AttentionConfig(
                hidden_size=HIDDEN_SIZE,
                num_attention_heads=NUM_ATTENTION_HEADS,
                num_key_value_heads=NUM_KEY_VALUE_HEADS)
            output = forward_fn(params, inputs, config)
            abserr = np.abs(jx2np(output) - pt2np(output_pt))
            print(f'torch - jax ({device})')
            print(f'- max: {abserr.max()}')
            print(f'- min: {abserr.min()}')
