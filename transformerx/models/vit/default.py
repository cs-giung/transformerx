"""Default configurations and utilities."""
from collections import OrderedDict

import jax
import jax.numpy as jnp
import torch

from transformers import ViTModel, ViTForImageClassification
from transformerx.models.vit.modeling import ViTConfig
from transformerx.typing import Pytree


PREDEFINED_CONFIGS = {
    'cs-giung/vit-large-patch16-imagenet21k': ViTConfig(
        hidden_size=1024,
        intermediate_size=4096,
        layer_norm_eps=1e-06,
        num_attention_heads=16,
        num_hidden_layers=24,
        num_labels=21843,
        patch_size=16,
        representation_size=1024,
    ),
    'cs-giung/vit-tiny-patch16-imagenet21k-augreg': ViTConfig(
        hidden_size=192,
        intermediate_size=768,
        layer_norm_eps=1e-06,
        num_attention_heads=3,
        num_hidden_layers=12,
        num_labels=21843,
        patch_size=16,
        representation_size=None,
    ),
    'cs-giung/vit-small-patch16-imagenet21k-augreg': ViTConfig(
        hidden_size=384,
        intermediate_size=1536,
        layer_norm_eps=1e-06,
        num_attention_heads=6,
        num_hidden_layers=12,
        num_labels=21843,
        patch_size=16,
        representation_size=None,
    ),
    'cs-giung/vit-base-patch16-imagenet21k-augreg': ViTConfig(
        hidden_size=768,
        intermediate_size=3072,
        layer_norm_eps=1e-06,
        num_attention_heads=12,
        num_hidden_layers=12,
        num_labels=21843,
        patch_size=16,
        representation_size=None,
    ),
    'cs-giung/vit-large-patch16-imagenet21k-augreg': ViTConfig(
        hidden_size=1024,
        intermediate_size=4096,
        layer_norm_eps=1e-06,
        num_attention_heads=16,
        num_hidden_layers=24,
        num_labels=21843,
        patch_size=16,
        representation_size=None,
    ),
}


def load_hf_params(model_name: str) -> OrderedDict:
    """Load pre-trained parameters from the Hugging Face Hub."""
    d1 = ViTModel.from_pretrained(
        model_name, torch_dtype=torch.float32).state_dict()
    d2 = ViTForImageClassification.from_pretrained(
        model_name, torch_dtype=torch.float32).state_dict()
    d2['vit.pooler.dense.bias'] = d1['pooler.dense.bias']
    d2['vit.pooler.dense.weight'] = d1['pooler.dense.weight']
    return d2


def load_jx_params(model_name: str) -> Pytree:
    """Returns pre-trained parameters."""
    return convert_hf_params_to_jx_params(
        load_hf_params(model_name), load_jx_config(model_name))


def load_jx_config(model_name: str) -> ViTConfig:
    """Returns pre-defined configurations."""
    return PREDEFINED_CONFIGS[model_name]


def convert_hf_params_to_jx_params(
        hf_params: OrderedDict, jx_config: ViTConfig) -> Pytree:
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
                    'vit.embeddings.position_embeddings'][0])}}
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
        pooler = {
            'weight': pt2jx(hf_params['vit.pooler.dense.weight']).T,
            'bias': pt2jx(hf_params['vit.pooler.dense.bias'])}
        head = {
            'weight': pt2jx(hf_params['classifier.weight']).T,
            'bias': pt2jx(hf_params['classifier.bias'])}

        params = {
            'embeddings': embeddings,
            'layers': layers,
            'post_layernorm': post_layernorm}

        if jx_config.representation_size:
            params['pooler'] = pooler

        if jx_config.num_labels:
            params['head'] = head

        return params
