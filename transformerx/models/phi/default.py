"""Default configurations and utilities."""
from collections import OrderedDict
from typing import Callable, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformerx.models.phi.modeling import PhiConfig
from transformerx.typing import Pytree


PREDEFINED_CONFIGS = {
    'microsoft/Phi-3-mini-4k-instruct': PhiConfig(
        hidden_size=3072,
        intermediate_size=8192,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=32,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        vocab_size=32064,
    ),
    'microsoft/Phi-3-medium-4k-instruct': PhiConfig(
        hidden_size=5120,
        intermediate_size=17920,
        num_attention_heads=40,
        num_hidden_layers=40,
        num_key_value_heads=10,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        vocab_size=32064,
    ),
}


def load_hf_params(model_name: str) -> OrderedDict:
    """Load pre-trained parameters from the Hugging Face Hub."""
    return AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).state_dict()


def load_jx_params(model_name: str) -> Pytree:
    """Returns pre-trained parameters."""
    return convert_hf_params_to_jx_params(
        load_hf_params(model_name), model_name)


def load_jx_config(model_name: str) -> PhiConfig:
    """Returns pre-defined configuration."""
    return PREDEFINED_CONFIGS[model_name]


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
        hf_params: OrderedDict, model_name: str) -> Pytree:
    """Converts pytorch state_dict in the transformerx format."""

    @torch.no_grad
    def pt2jx(e):
        return jnp.asarray(e.cpu().numpy())

    # given our use of pre-trained model, flexibility might not be crucial.
    num_hidden_layers = 1 + max(
        int(e.split('.')[2]) for e in hf_params.keys()
        if e.startswith('model.layers.'))
    if model_name == 'microsoft/Phi-3-mini-4k-instruct':
        qkv2q = lambda e: e[   0:3072]
        qkv2k = lambda e: e[3072:6144]
        qkv2v = lambda e: e[6144:9216]
        gu2g = lambda e: e[    0:8192]
        gu2u = lambda e: e[8192:16384]
    if model_name == 'microsoft/Phi-3-medium-4k-instruct':
        qkv2q = lambda e: e[   0:5120]
        qkv2k = lambda e: e[5120:6400]
        qkv2v = lambda e: e[6400:7680]
        gu2g = lambda e: e[    0:17920]
        gu2u = lambda e: e[17920:35840]

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
                    'q_proj': {'weight': qkv2q(pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.qkv_proj.weight'])).T},
                    'k_proj': {'weight': qkv2k(pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.qkv_proj.weight'])).T},
                    'v_proj': {'weight': qkv2v(pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.qkv_proj.weight'])).T},
                    'o_proj': {'weight': pt2jx(hf_params[
                        f'model.layers.{i}.self_attn.o_proj.weight']).T}},
                'post_attention_layernorm': {
                    'weight': pt2jx(hf_params[
                        f'model.layers.{i}.post_attention_layernorm.weight'])},
                'mlp': {
                    'g_proj': {'weight': gu2g(pt2jx(hf_params[
                        f'model.layers.{i}.mlp.gate_up_proj.weight'])).T},
                    'u_proj': {'weight': gu2u(pt2jx(hf_params[
                        f'model.layers.{i}.mlp.gate_up_proj.weight'])).T},
                    'd_proj': {'weight': pt2jx(hf_params[
                        f'model.layers.{i}.mlp.down_proj.weight']).T}}
            } for i in range(num_hidden_layers)}
        lm_head = {
            'weight': pt2jx(hf_params['lm_head.weight']).T}

        return {
            'embed_tokens': embed_tokens,
            'layers': layers,
            'norm': norm,
            'lm_head': lm_head,
        }
