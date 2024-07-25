"""Default configurations and utilities."""
from collections import OrderedDict
from typing import Callable, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformerx.models.llama.modeling import LlamaConfig
from transformerx.typing import Pytree


PREDEFINED_CONFIGS = {
    'huggyllama/llama-7b': LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=32,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        sliding_window=None,
        vocab_size=32000,
    ),
    'huggyllama/llama-13b': LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_attention_heads=40,
        num_hidden_layers=40,
        num_key_value_heads=40,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        sliding_window=None,
        vocab_size=32000,
    ),
    'huggyllama/llama-30b': LlamaConfig(
        hidden_size=6656,
        intermediate_size=17920,
        num_attention_heads=52,
        num_hidden_layers=60,
        num_key_value_heads=52,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        sliding_window=None,
        vocab_size=32000,
    ),
    'huggyllama/llama-65b': LlamaConfig(
        hidden_size=8192,
        intermediate_size=22016,
        num_attention_heads=64,
        num_hidden_layers=80,
        num_key_value_heads=64,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        sliding_window=None,
        vocab_size=32000,
    ),
    'meta-llama/Llama-2-7b-hf': LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=32,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        sliding_window=None,
        vocab_size=32000,
    ),
    'meta-llama/Llama-2-7b-chat-hf': LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=32,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        sliding_window=None,
        vocab_size=32000,
    ),
    'meta-llama/Llama-2-13b-hf': LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_attention_heads=40,
        num_hidden_layers=40,
        num_key_value_heads=40,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        sliding_window=None,
        vocab_size=32000,
    ),
    'meta-llama/Llama-2-13b-chat-hf': LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_attention_heads=40,
        num_hidden_layers=40,
        num_key_value_heads=40,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        sliding_window=None,
        vocab_size=32000,
    ),
    'meta-llama/Llama-2-70b-hf': LlamaConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_attention_heads=64,
        num_hidden_layers=80,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        sliding_window=None,
        vocab_size=32000,
    ),
    'meta-llama/Llama-2-70b-chat-hf': LlamaConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_attention_heads=64,
        num_hidden_layers=80,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        sliding_window=None,
        vocab_size=32000,
    ),
    'meta-llama/Meta-Llama-3-8B': LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=500000.0,
        sliding_window=None,
        vocab_size=128256,
    ),
    'meta-llama/Meta-Llama-3-8B-Instruct': LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=500000.0,
        sliding_window=None,
        vocab_size=128256,
    ),
    'meta-llama/Meta-Llama-3-70B': LlamaConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_attention_heads=64,
        num_hidden_layers=80,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=500000.0,
        sliding_window=None,
        vocab_size=128256,
    ),
    'meta-llama/Meta-Llama-3.1-8B': LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=500000.0,
        sliding_window=None,
        vocab_size=128256,
    ),
    'meta-llama/Meta-Llama-3.1-8B-Instruct': LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=500000.0,
        sliding_window=None,
        vocab_size=128256,
    ),
    'mistralai/Mistral-7B-v0.1': LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        sliding_window=4096,
        vocab_size=32768,
    ),
    'mistral-community/Mistral-7B-v0.2': LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=1000000.0,
        sliding_window=None,
        vocab_size=32000,
    ),
    'mistralai/Mistral-7B-v0.3': LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=1000000.0,
        sliding_window=None,
        vocab_size=32768,
    ),
    'mistralai/Mistral-7B-Instruct-v0.1': LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        sliding_window=4096,
        vocab_size=32768,
    ),
    'mistralai/Mistral-7B-Instruct-v0.2': LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=1000000.0,
        sliding_window=None,
        vocab_size=32000,
    ),
    'mistralai/Mistral-7B-Instruct-v0.3': LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=1000000.0,
        sliding_window=None,
        vocab_size=32768,
    ),
    'microsoft/Phi-3-mini-4k-instruct': LlamaConfig(
        hidden_size=3072,
        intermediate_size=8192,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=32,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        sliding_window=2047,
        vocab_size=32064,
    ),
    'microsoft/Phi-3-medium-4k-instruct': LlamaConfig(
        hidden_size=5120,
        intermediate_size=17920,
        num_attention_heads=40,
        num_hidden_layers=40,
        num_key_value_heads=10,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        sliding_window=2047,
        vocab_size=32064,
    ),
}


def load_hf_params(model_name: str) -> OrderedDict:
    """Load pre-trained parameters from the Hugging Face Hub."""
    return AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).state_dict()


def load_jx_config(model_name: str) -> LlamaConfig:
    """Returns pre-defined configuration."""
    return PREDEFINED_CONFIGS[model_name]


def load_jx_params(model_name: str) -> Pytree:
    """Returns pre-trained parameters."""
    return convert_hf_params_to_jx_params(
        load_hf_params(model_name), load_jx_config(model_name), model_name)


def get_tokenize_fn(
        model_name: str,
        *,
        max_length: int,
        add_special_tokens: bool = True,
        padding_side: str = 'right',
        return_tensors: str = 'np',
    ) -> Callable:
    """Returns customized tokenization function."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if padding_side not in ['right', 'left']:
        raise AssertionError('padding_side should be `right` or `left`.')
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = padding_side

    def tokenize_fn(prompt: Union[str, List[str]]):
        batch_encoding = tokenizer(
            prompt, add_special_tokens=add_special_tokens,
            padding='max_length', truncation=True,
            max_length=max_length, return_tensors='np')

        input_ids = batch_encoding['input_ids']
        attention_mask = batch_encoding['attention_mask']

        if padding_side == 'right':
            bos_idx = np.zeros((attention_mask.shape[0]))
            eos_idx = np.sum(attention_mask, axis=-1)
        elif padding_side == 'left':
            bos_idx = max_length - np.sum(attention_mask, axis=-1)
            eos_idx = max_length * np.ones((attention_mask.shape[0]))
        else:
            raise AssertionError(f'Unknown padding_side={padding_side}')
        bos_idx, eos_idx = bos_idx.astype(int), eos_idx.astype(int)

        position_ids = np.zeros_like(attention_mask)
        for i in range(attention_mask.shape[0]):
            position_ids[
                i, bos_idx[i]:eos_idx[i]] = np.arange(eos_idx[i] - bos_idx[i])

        tokenized = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids}

        if return_tensors == 'jax':
            tokenized = jax.tree_util.tree_map(jnp.array, tokenized)

        return tokenized

    return tokenize_fn


def convert_hf_params_to_jx_params(
        hf_params: OrderedDict,
        jx_config: LlamaConfig,
        model_name: str,
    ) -> Pytree:
    """Converts pytorch state_dict in the transformerx format."""

    @torch.no_grad
    def pt2jx(e):
        return jnp.asarray(e.cpu().numpy())

    device = jax.devices('cpu')[0]
    with jax.default_device(device):
        embed_tokens = {
            'weight': pt2jx(hf_params['model.embed_tokens.weight'])}
        norm = {
            'weight': pt2jx(hf_params['model.norm.weight'])}
        layers = {}
        for i in range(jx_config.num_hidden_layers):
            layers[f'{i}'] = {}
            layers[f'{i}']['input_layernorm'] = {
                'weight': pt2jx(hf_params[
                    f'model.layers.{i}.input_layernorm.weight'])}
            layers[f'{i}']['post_attention_layernorm'] = {
                'weight': pt2jx(hf_params[
                    f'model.layers.{i}.post_attention_layernorm.weight'])}

            layers[f'{i}']['self_attn'] = {}
            layers[f'{i}']['mlp'] = {}
            if model_name == 'microsoft/Phi-3-mini-4k-instruct':
                layers[f'{i}']['self_attn']['q_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.qkv_proj.weight'
                    ][0:3072]).T}
                layers[f'{i}']['self_attn']['k_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.qkv_proj.weight'
                    ][3072:6144]).T}
                layers[f'{i}']['self_attn']['v_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.qkv_proj.weight'
                    ][6144:9216]).T}
                layers[f'{i}']['mlp']['g_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.mlp.gate_up_proj.weight'
                    ][:8192]).T}
                layers[f'{i}']['mlp']['u_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.mlp.gate_up_proj.weight'
                    ][8192:]).T}

            elif model_name == 'microsoft/Phi-3-medium-4k-instruct':
                layers[f'{i}']['self_attn']['q_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.qkv_proj.weight'
                    ][0:5120]).T}
                layers[f'{i}']['self_attn']['k_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.qkv_proj.weight'
                    ][5120:6400]).T}
                layers[f'{i}']['self_attn']['v_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.qkv_proj.weight'
                    ][6400:7680]).T}
                layers[f'{i}']['mlp']['g_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.mlp.gate_up_proj.weight'
                    ][:17920]).T}
                layers[f'{i}']['mlp']['u_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.mlp.gate_up_proj.weight'
                    ][17920:]).T}

            else:
                layers[f'{i}']['self_attn']['q_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.q_proj.weight']).T}
                layers[f'{i}']['self_attn']['k_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.k_proj.weight']).T}
                layers[f'{i}']['self_attn']['v_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.v_proj.weight']).T}
                layers[f'{i}']['mlp']['g_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.mlp.gate_proj.weight']).T}
                layers[f'{i}']['mlp']['u_proj'] = {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.mlp.up_proj.weight']).T}

            layers[f'{i}']['self_attn']['o_proj'] = {
                'weight': pt2jx(hf_params[
                    f'model.layers.{i}.self_attn.o_proj.weight']).T}
            layers[f'{i}']['mlp']['d_proj'] = {
                'weight': pt2jx(hf_params[
                    f'model.layers.{i}.mlp.down_proj.weight']).T}

        lm_head = {
            'weight': pt2jx(hf_params['lm_head.weight']).T}

        return {
            'embed_tokens': embed_tokens,
            'layers': layers,
            'norm': norm,
            'lm_head': lm_head,
        }
