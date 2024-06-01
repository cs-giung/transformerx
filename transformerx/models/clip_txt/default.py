"""Default configurations and utilities."""
from collections import OrderedDict
from typing import Callable, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch

from transformers import AutoTokenizer, CLIPModel
from transformerx.models.clip_txt.modeling import CLIPTxTConfig
from transformerx.typing import Pytree


PREDEFINED_CONFIGS = {
    'openai/clip-vit-base-patch32': CLIPTxTConfig(
        hidden_act='quick_gelu',
        hidden_size=512,
        intermediate_size=2048,
        layer_norm_eps=1e-05,
        num_attention_heads=8,
        num_hidden_layers=12,
        projection_dim=512,
        vocab_size=49408,
    ),
    'openai/clip-vit-base-patch16': CLIPTxTConfig(
        hidden_act='quick_gelu',
        hidden_size=512,
        intermediate_size=2048,
        layer_norm_eps=1e-05,
        num_attention_heads=8,
        num_hidden_layers=12,
        projection_dim=512,
        vocab_size=49408,
    ),
    'openai/clip-vit-large-patch14': CLIPTxTConfig(
        hidden_act='quick_gelu',
        hidden_size=768,
        intermediate_size=3072,
        layer_norm_eps=1e-05,
        num_attention_heads=12,
        num_hidden_layers=12,
        projection_dim=768,
        vocab_size=49408,
    ),
    'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': CLIPTxTConfig(
        hidden_act='gelu',
        hidden_size=1024,
        intermediate_size=4096,
        layer_norm_eps=1e-05,
        num_attention_heads=16,
        num_hidden_layers=24,
        projection_dim=1024,
        vocab_size=49408,
    ),
    'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k': CLIPTxTConfig(
        hidden_act='gelu',
        hidden_size=1280,
        intermediate_size=5120,
        layer_norm_eps=1e-05,
        num_attention_heads=20,
        num_hidden_layers=32,
        projection_dim=1280,
        vocab_size=49408,
    ),
}


def load_hf_params(model_name: str) -> OrderedDict:
    """Load pre-trained parameters from the Hugging Face Hub."""
    return CLIPModel.from_pretrained(
        model_name, torch_dtype=torch.float32).state_dict()


def load_jx_config(model_name: str) -> CLIPTxTConfig:
    """Returns pre-defined configurations."""
    return PREDEFINED_CONFIGS[model_name]


def load_jx_params(model_name: str) -> Pytree:
    """Returns pre-trained parameters."""
    return convert_hf_params_to_jx_params(
        load_hf_params(model_name), load_jx_config(model_name))


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
        hf_params: OrderedDict, jx_config: CLIPTxTConfig) -> Pytree:
    """Converts pytorch state_dict in the transformerx format."""

    @torch.no_grad
    def pt2jx(e):
        return jnp.asarray(e.cpu().numpy())

    device = jax.devices('cpu')[0]
    with jax.default_device(device):
        embeddings = {
            'token_embedding': {
                'weight': pt2jx(hf_params[
                    'text_model.embeddings.token_embedding.weight'])},
            'position_embedding': {
                'weight': pt2jx(hf_params[
                    'text_model.embeddings.position_embedding.weight'])}}
        layers = {
            f'{i}': {
                'pre_layernorm': {
                    'weight': pt2jx(hf_params[
                        f'text_model.encoder.layers.{i}.'
                        f'layer_norm1.weight']),
                    'bias': pt2jx(hf_params[
                        f'text_model.encoder.layers.{i}.'
                        f'layer_norm1.bias'])},
                'post_layernorm': {
                    'weight': pt2jx(hf_params[
                        f'text_model.encoder.layers.{i}.'
                        f'layer_norm2.weight']),
                    'bias': pt2jx(hf_params[
                        f'text_model.encoder.layers.{i}.'
                        f'layer_norm2.bias'])},
                'mlp': {
                    'u_proj': {
                        'weight': pt2jx(hf_params[
                            f'text_model.encoder.layers.{i}.'
                            f'mlp.fc1.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'text_model.encoder.layers.{i}.'
                            f'mlp.fc1.bias'])},
                    'd_proj': {
                        'weight': pt2jx(hf_params[
                            f'text_model.encoder.layers.{i}.'
                            f'mlp.fc2.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'text_model.encoder.layers.{i}.'
                            f'mlp.fc2.bias'])}},
                'self_attn': {
                    'q_proj': {
                        'weight': pt2jx(hf_params[
                            f'text_model.encoder.layers.{i}.'
                            f'self_attn.q_proj.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'text_model.encoder.layers.{i}.'
                            f'self_attn.q_proj.bias'])},
                    'k_proj': {
                        'weight': pt2jx(hf_params[
                            f'text_model.encoder.layers.{i}.'
                            f'self_attn.k_proj.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'text_model.encoder.layers.{i}.'
                            f'self_attn.k_proj.bias'])},
                    'v_proj': {
                        'weight': pt2jx(hf_params[
                            f'text_model.encoder.layers.{i}.'
                            f'self_attn.v_proj.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'text_model.encoder.layers.{i}.'
                            f'self_attn.v_proj.bias'])},
                    'o_proj': {
                        'weight': pt2jx(hf_params[
                            f'text_model.encoder.layers.{i}.'
                            f'self_attn.out_proj.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'text_model.encoder.layers.{i}.'
                            f'self_attn.out_proj.bias'])}},
            } for i in range(jx_config.num_hidden_layers)}
        final_layer_norm = {
            'weight': pt2jx(hf_params['text_model.final_layer_norm.weight']),
            'bias': pt2jx(hf_params['text_model.final_layer_norm.bias'])}
        projection = {
            'weight': pt2jx(hf_params['text_projection.weight'].T)}

        params = {
            'embeddings': embeddings,
            'layers': layers,
            'final_layer_norm': final_layer_norm,
            'projection': projection}

        return params
