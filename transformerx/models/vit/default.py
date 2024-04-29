"""Default configurations and utilities for the ViTmodel."""
from collections import OrderedDict

import jax
import jax.numpy as jnp
import torch

from transformers import ViTForImageClassification
from transformerx.models.vit.modeling import ViTConfig
from transformerx.typing import Pytree


PREDEFINED_CONFIGS = {
    'WinKawaks/vit-tiny-patch16-224': ViTConfig(
        hidden_size=192,
        intermediate_size=768,
        num_attention_heads=3,
        num_hidden_layers=12,
        patch_size=16,
        layer_norm_eps=1e-12,
    ),
    'WinKawaks/vit-small-patch16-224': ViTConfig(
        hidden_size=384,
        intermediate_size=1536,
        num_attention_heads=6,
        num_hidden_layers=12,
        patch_size=16,
        layer_norm_eps=1e-12,
    ),
    'google/vit-base-patch16-224': ViTConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        patch_size=16,
        layer_norm_eps=1e-12,
    ),
    'google/vit-large-patch16-224': ViTConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        num_hidden_layers=24,
        patch_size=16,
        layer_norm_eps=1e-12,
    ),
}


def load_hf_params(model_name: str) -> OrderedDict:
    """Load pre-trained parameters from the Hugging Face Hub."""
    return ViTForImageClassification.from_pretrained(
        model_name, torch_dtype=torch.float32).state_dict()


def load_jx_params(model_name: str) -> Pytree:
    """Returns pre-trained parameters."""
    return convert_hf_params_to_jx_params(load_hf_params(model_name))


def load_jx_config(model_name: str) -> ViTConfig:
    """Returns pre-defined configurations."""
    return PREDEFINED_CONFIGS[model_name]


def convert_hf_params_to_jx_params(hf_params: OrderedDict) -> Pytree:
    """Converts pytorch state_dict in the transformerx format."""

    @torch.no_grad
    def pt2jx(e):
        return jnp.asarray(e.cpu().numpy())

    # given our use of pre-trained model, flexibility might not be crucial.
    num_hidden_layers = 1 + max(
        int(e.split('.')[3]) for e in hf_params.keys()
        if e.startswith('vit.encoder.layer.'))
    hidden_size = hf_params['vit.embeddings.cls_token'].shape[-1]

    device = jax.devices('cpu')[0]
    with jax.default_device(device):
        embeddings = {
            'class_embedding': {
                'weight': pt2jx(hf_params['vit.embeddings.cls_token'][0])},
            'patch_embedding': {
                'weight': pt2jx(hf_params[
                    'vit.embeddings.patch_embeddings.projection.weight'
                    ]).reshape(hidden_size, -1).T,
                'bias': pt2jx(hf_params[
                    'vit.embeddings.patch_embeddings.projection.bias'])},
            'position_embedding': {
                'weight': pt2jx(hf_params[
                    'vit.embeddings.position_embeddings'])}}
        layers = {
            f'{i}': {
                'pre_layernorm': {
                    'weight': pt2jx(hf_params[
                        f'vit.encoder.layer.{i}.'
                        f'layernorm_before.weight']),
                    'bias': pt2jx(hf_params[
                        f'vit.encoder.layer.{i}.'
                        f'layernorm_before.bias'])},
                'post_layernorm': {
                    'weight': pt2jx(hf_params[
                        f'vit.encoder.layer.{i}.'
                        f'layernorm_after.weight']),
                    'bias': pt2jx(hf_params[
                        f'vit.encoder.layer.{i}.'
                        f'layernorm_after.bias'])},
                'mlp': {
                    'u_proj': {
                        'weight': pt2jx(hf_params[
                            f'vit.encoder.layer.{i}.'
                            f'intermediate.dense.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'vit.encoder.layer.{i}.'
                            f'intermediate.dense.bias'])},
                    'd_proj': {
                        'weight': pt2jx(hf_params[
                            f'vit.encoder.layer.{i}.'
                            f'output.dense.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'vit.encoder.layer.{i}.'
                            f'output.dense.bias'])}},
                'self_attn': {
                    'q_proj': {
                        'weight': pt2jx(hf_params[
                            f'vit.encoder.layer.{i}.'
                            f'attention.attention.query.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'vit.encoder.layer.{i}.'
                            f'attention.attention.query.bias'])},
                    'k_proj': {
                        'weight': pt2jx(hf_params[
                            f'vit.encoder.layer.{i}.'
                            f'attention.attention.key.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'vit.encoder.layer.{i}.'
                            f'attention.attention.key.bias'])},
                    'v_proj': {
                        'weight': pt2jx(hf_params[
                            f'vit.encoder.layer.{i}.'
                            f'attention.attention.value.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'vit.encoder.layer.{i}.'
                            f'attention.attention.value.bias'])},
                    'o_proj': {
                        'weight': pt2jx(hf_params[
                            f'vit.encoder.layer.{i}.'
                            f'attention.output.dense.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'vit.encoder.layer.{i}.'
                            f'attention.output.dense.bias'])}},
            } for i in range(num_hidden_layers)}
        post_layernorm = {
            'weight': pt2jx(hf_params['vit.layernorm.weight']),
            'bias': pt2jx(hf_params['vit.layernorm.bias'])}
        head = {
            'weight': pt2jx(hf_params['classifier.weight']).T,
            'bias': pt2jx(hf_params['classifier.bias'])}

    return {
        'embeddings': embeddings,
        'layers': layers,
        'post_layernorm': post_layernorm,
        'head': head,
    }
