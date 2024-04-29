"""Default configurations and utilities for the CLIP model."""
from collections import OrderedDict

import jax
import jax.numpy as jnp
import torch

from transformers import CLIPModel
from transformerx.models.clip_vision.modeling import CLIPVisionConfig
from transformerx.typing import Pytree


PREDEFINED_CONFIGS = {
    'openai/clip-vit-base-patch32': CLIPVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        patch_size=32,
        projection_dim=512,
        layer_norm_eps=1e-05,
    ),
    'openai/clip-vit-base-patch16': CLIPVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        patch_size=16,
        projection_dim=512,
        layer_norm_eps=1e-05,
    ),
    'openai/clip-vit-large-patch14': CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        num_hidden_layers=24,
        patch_size=14,
        projection_dim=768,
        layer_norm_eps=1e-05,
    ),
}


def load_hf_params(model_name: str) -> OrderedDict:
    """Load pre-trained parameters from the Hugging Face Hub."""
    return CLIPModel.from_pretrained(
        model_name, torch_dtype=torch.float32).state_dict()


def load_jx_params(model_name: str) -> Pytree:
    """Returns pre-trained parameters."""
    return convert_hf_params_to_jx_params(load_hf_params(model_name))


def load_jx_config(model_name: str) -> CLIPVisionConfig:
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
        if e.startswith('vision_model.encoder.layers.'))
    hidden_size = hf_params[
        'vision_model.embeddings.class_embedding'].shape[0]

    device = jax.devices('cpu')[0]
    with jax.default_device(device):
        embeddings = {
            'class_embedding': {
                'weight': pt2jx(hf_params[
                    'vision_model.embeddings.class_embedding'][None])},
            'patch_embedding': {
                'weight': pt2jx(hf_params[
                    'vision_model.embeddings.patch_embedding.weight'
                ]).reshape(hidden_size, -1).T},
            'position_embedding': {
                'weight': pt2jx(hf_params[
                    'vision_model.embeddings.position_embedding.weight'])}}
        pre_layernorm = {
            'weight': pt2jx(hf_params['vision_model.pre_layrnorm.weight']),
            'bias': pt2jx(hf_params['vision_model.pre_layrnorm.bias'])}
        layers = {
            f'{i}': {
                'pre_layernorm': {
                    'weight': pt2jx(hf_params[
                        f'vision_model.encoder.layers.{i}.'
                        f'layer_norm1.weight']),
                    'bias': pt2jx(hf_params[
                        f'vision_model.encoder.layers.{i}.'
                        f'layer_norm1.bias'])},
                'post_layernorm': {
                    'weight': pt2jx(hf_params[
                        f'vision_model.encoder.layers.{i}.'
                        f'layer_norm2.weight']),
                    'bias': pt2jx(hf_params[
                        f'vision_model.encoder.layers.{i}.'
                        f'layer_norm2.bias'])},
                'mlp': {
                    'u_proj': {
                        'weight': pt2jx(hf_params[
                            f'vision_model.encoder.layers.{i}.'
                            f'mlp.fc1.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'vision_model.encoder.layers.{i}.'
                            f'mlp.fc1.bias'])},
                    'd_proj': {
                        'weight': pt2jx(hf_params[
                            f'vision_model.encoder.layers.{i}.'
                            f'mlp.fc2.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'vision_model.encoder.layers.{i}.'
                            f'mlp.fc2.bias'])}},
                'self_attn': {
                    'q_proj': {
                        'weight': pt2jx(hf_params[
                            f'vision_model.encoder.layers.{i}.'
                            f'self_attn.q_proj.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'vision_model.encoder.layers.{i}.'
                            f'self_attn.q_proj.bias'])},
                    'k_proj': {
                        'weight': pt2jx(hf_params[
                            f'vision_model.encoder.layers.{i}.'
                            f'self_attn.k_proj.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'vision_model.encoder.layers.{i}.'
                            f'self_attn.k_proj.bias'])},
                    'v_proj': {
                        'weight': pt2jx(hf_params[
                            f'vision_model.encoder.layers.{i}.'
                            f'self_attn.v_proj.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'vision_model.encoder.layers.{i}.'
                            f'self_attn.v_proj.bias'])},
                    'o_proj': {
                        'weight': pt2jx(hf_params[
                            f'vision_model.encoder.layers.{i}.'
                            f'self_attn.out_proj.weight'].T),
                        'bias': pt2jx(hf_params[
                            f'vision_model.encoder.layers.{i}.'
                            f'self_attn.out_proj.bias'])}},
            } for i in range(num_hidden_layers)}
        post_layernorm = {
            'weight': pt2jx(hf_params['vision_model.post_layernorm.weight']),
            'bias': pt2jx(hf_params['vision_model.post_layernorm.bias'])}
        projection = {
            'weight': pt2jx(hf_params['visual_projection.weight'].T)}

    return {
        'embeddings': embeddings,
        'pre_layernorm': pre_layernorm,
        'layers': layers,
        'post_layernorm': post_layernorm,
        'projection': projection,
    }
