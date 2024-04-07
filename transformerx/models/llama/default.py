"""Default configurations and utilities for the Llama model."""
from collections import OrderedDict

import jax
import jax.numpy as jnp
import torch

from transformerx.models.llama.modeling import LlamaConfig, LlamaParams


PREDEFINED_CONFIGS = {
    'meta-llama/Llama-2-7b-hf': LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=32,
        rms_norm_eps=1e-05,
    ),
    'meta-llama/Llama-2-13b-hf': LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_attention_heads=40,
        num_hidden_layers=40,
        num_key_value_heads=40,
        rms_norm_eps=1e-05,
    ),
}


def convert_hf_params_to_jx_params(hf_params: OrderedDict) -> LlamaParams:
    """Converts pytorch state_dict in the transformerx format."""

    @torch.no_grad
    def pt2jx(e):
        return jnp.asarray(e.cpu().numpy())

    # given our use of pre-trained model, flexibility might not be crucial.
    num_hidden_layers = 1 + max(
        int(e.split('.')[2]) for e in hf_params.keys()
        if e.startswith('model.layers.'))

    device = jax.devices('cpu')[0]
    with jax.default_device(device):
        embed_tokens = {
            'weight': pt2jx(hf_params['model.embed_tokens.weight'])}
        norm = {
            'weight': pt2jx(hf_params['model.norm.weight'])}
        layers = {
            f'{i}': {
                'input_layernorm': {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.input_layernorm.weight'])},
                'self_attn': {
                    'q_proj': {'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.q_proj.weight']).T},
                    'k_proj': {'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.k_proj.weight']).T},
                    'v_proj': {'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.v_proj.weight']).T},
                    'o_proj': {'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.o_proj.weight']).T}},
                'post_attention_layernorm': {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.post_attention_layernorm.weight'])},
                'mlp': {
                    'g_proj': {'weight': pt2jx(hf_params[
                        f'model.layers.{i}.mlp.gate_proj.weight']).T},
                    'u_proj': {'weight': pt2jx(hf_params[
                        f'model.layers.{i}.mlp.up_proj.weight']).T},
                    'd_proj': {'weight': pt2jx(hf_params[
                        f'model.layers.{i}.mlp.down_proj.weight']).T}}
            } for i in range(num_hidden_layers)}

        return LlamaParams(
            embed_tokens=embed_tokens, layers=layers, norm=norm)
