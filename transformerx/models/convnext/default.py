"""Default configurations and utilities."""
from collections import OrderedDict

import jax
import jax.numpy as jnp
import torch

from transformers import \
    ConvNextForImageClassification, ConvNextV2ForImageClassification
from transformerx.models.convnext.modeling import ConvNextConfig
from transformerx.typing import Pytree


PREDEFINED_CONFIGS = {
    'cs-giung/convnext-v1-tiny-imagenet1k': ConvNextConfig(
        grn_eps=None,
        hidden_sizes=(96, 192, 384, 768),
        layer_norm_eps=1e-06,
        num_hidden_layers=(3, 3, 9, 3),
        num_labels=1000,
        patch_size=4,
        post_layernorm=True,
    ),
    'cs-giung/convnext-v1-small-imagenet1k': ConvNextConfig(
        grn_eps=None,
        hidden_sizes=(96, 192, 384, 768),
        layer_norm_eps=1e-06,
        num_hidden_layers=(3, 3, 27, 3),
        num_labels=1000,
        patch_size=4,
        post_layernorm=True,
    ),
    'cs-giung/convnext-v1-base-imagenet1k': ConvNextConfig(
        grn_eps=None,
        hidden_sizes=(128, 256, 512, 1024),
        layer_norm_eps=1e-06,
        num_hidden_layers=(3, 3, 27, 3),
        num_labels=1000,
        patch_size=4,
        post_layernorm=True,
    ),
    'cs-giung/convnext-v1-large-imagenet1k': ConvNextConfig(
        grn_eps=None,
        hidden_sizes=(192, 384, 768, 1536),
        layer_norm_eps=1e-06,
        num_hidden_layers=(3, 3, 27, 3),
        num_labels=1000,
        patch_size=4,
        post_layernorm=True,
    ),
    'cs-giung/convnext-v1-tiny-imagenet21k': ConvNextConfig(
        grn_eps=None,
        hidden_sizes=(96, 192, 384, 768),
        layer_norm_eps=1e-06,
        num_hidden_layers=(3, 3, 9, 3),
        num_labels=21843,
        patch_size=4,
        post_layernorm=True,
    ),
    'cs-giung/convnext-v1-small-imagenet21k': ConvNextConfig(
        grn_eps=None,
        hidden_sizes=(96, 192, 384, 768),
        layer_norm_eps=1e-06,
        num_hidden_layers=(3, 3, 27, 3),
        num_labels=21843,
        patch_size=4,
        post_layernorm=True,
    ),
    'cs-giung/convnext-v1-base-imagenet21k': ConvNextConfig(
        grn_eps=None,
        hidden_sizes=(128, 256, 512, 1024),
        layer_norm_eps=1e-06,
        num_hidden_layers=(3, 3, 27, 3),
        num_labels=21843,
        patch_size=4,
        post_layernorm=True,
    ),
    'cs-giung/convnext-v1-large-imagenet21k': ConvNextConfig(
        grn_eps=None,
        hidden_sizes=(192, 384, 768, 1536),
        layer_norm_eps=1e-06,
        num_hidden_layers=(3, 3, 27, 3),
        num_labels=21843,
        patch_size=4,
        post_layernorm=True,
    ),
    'cs-giung/convnext-v1-xlarge-imagenet21k': ConvNextConfig(
        grn_eps=None,
        hidden_sizes=(256, 512, 1024, 2048),
        layer_norm_eps=1e-06,
        num_hidden_layers=(3, 3, 27, 3),
        num_labels=21843,
        patch_size=4,
        post_layernorm=True,
    ),
    'cs-giung/convnext-v2-base-imagenet1k-fcmae': ConvNextConfig(
        grn_eps=1e-06,
        hidden_sizes=(128, 256, 512, 1024),
        layer_norm_eps=1e-06,
        num_hidden_layers=(3, 3, 27, 3),
        num_labels=None,
        patch_size=4,
        post_layernorm=False,
    ),
    'cs-giung/convnext-v2-large-imagenet1k-fcmae': ConvNextConfig(
        grn_eps=1e-06,
        hidden_sizes=(192, 384, 768, 1536),
        layer_norm_eps=1e-06,
        num_hidden_layers=(3, 3, 27, 3),
        num_labels=None,
        patch_size=4,
        post_layernorm=False,
    ),
    'cs-giung/convnext-v2-huge-imagenet1k-fcmae': ConvNextConfig(
        grn_eps=1e-06,
        hidden_sizes=(352, 704, 1408, 2816),
        layer_norm_eps=1e-06,
        num_hidden_layers=(3, 3, 27, 3),
        num_labels=None,
        patch_size=4,
        post_layernorm=False,
    ),
}


def load_hf_params(model_name: str) -> OrderedDict:
    """Load pre-trained parameters from the Hugging Face Hub."""
    if 'v1' in model_name:
        return ConvNextForImageClassification.from_pretrained(
            model_name, torch_dtype=torch.float32).state_dict()
    if 'v2' in model_name:
        return ConvNextV2ForImageClassification.from_pretrained(
            model_name, torch_dtype=torch.float32).state_dict()


def load_jx_params(model_name: str) -> Pytree:
    """Returns pre-trained parameters."""
    return convert_hf_params_to_jx_params(
        load_hf_params(model_name), load_jx_config(model_name))


def load_jx_config(model_name: str) -> ConvNextConfig:
    """Returns pre-defined configurations."""
    return PREDEFINED_CONFIGS[model_name]


def convert_hf_params_to_jx_params(
        hf_params: OrderedDict, jx_config: ConvNextConfig) -> Pytree:
    """Converts pytorch state_dict in the transformerx format."""

    @torch.no_grad
    def pt2jx(e):
        return jnp.asarray(e.cpu().numpy())

    prefix = 'convnextv2' if jx_config.grn_eps is not None else 'convnext'
    device = jax.devices('cpu')[0]
    with jax.default_device(device):
        embeddings = {
            'patch_embedding': {
                'weight': pt2jx(hf_params[
                    f'{prefix}.embeddings.patch_embeddings.weight'
                    ]).reshape(jx_config.hidden_sizes[0], -1).T,
                'bias': pt2jx(hf_params[
                    f'{prefix}.embeddings.patch_embeddings.bias'])}}
        pre_layernorm = {
            'weight': pt2jx(hf_params[f'{prefix}.embeddings.layernorm.weight']),
            'bias': pt2jx(hf_params[f'{prefix}.embeddings.layernorm.bias'])}
        layers = {}
        for i, num_hidden_layers in enumerate(jx_config.num_hidden_layers):
            layers[f'{i}'] = {}
            for j in range(num_hidden_layers):
                layers[f'{i}'][f'{j}'] = {}
                if i > 0:
                    layers[f'{i}'][f'{j}']['dsnorm'] = {
                        'weight': pt2jx(hf_params[
                            f'{prefix}.encoder.stages.{i}.downsampling_layer.'
                            f'0.weight']),
                        'bias': pt2jx(hf_params[
                            f'{prefix}.encoder.stages.{i}.downsampling_layer.'
                            f'0.bias'])}
                    layers[f'{i}'][f'{j}']['dsconv'] = {
                        'weight': pt2jx(hf_params[
                            f'{prefix}.encoder.stages.{i}.downsampling_layer.'
                            f'1.weight']).transpose(2, 3, 1, 0),
                        'bias': pt2jx(hf_params[
                            f'{prefix}.encoder.stages.{i}.downsampling_layer.'
                            f'1.bias'])}
                layers[f'{i}'][f'{j}']['dwconv'] = {
                    'weight': pt2jx(hf_params[
                        f'{prefix}.encoder.stages.{i}.layers.{j}.'
                        f'dwconv.weight']).transpose(2, 3, 1, 0),
                    'bias': pt2jx(hf_params[
                        f'{prefix}.encoder.stages.{i}.layers.{j}.'
                        f'dwconv.bias'])}
                layers[f'{i}'][f'{j}']['layernorm'] = {
                    'weight': pt2jx(hf_params[
                        f'{prefix}.encoder.stages.{i}.layers.{j}.'
                        f'layernorm.weight']),
                    'bias': pt2jx(hf_params[
                        f'{prefix}.encoder.stages.{i}.layers.{j}.'
                        f'layernorm.bias'])}
                layers[f'{i}'][f'{j}']['pwconv1'] = {
                    'weight': pt2jx(hf_params[
                        f'{prefix}.encoder.stages.{i}.layers.{j}.'
                        f'pwconv1.weight'].T),
                    'bias': pt2jx(hf_params[
                        f'{prefix}.encoder.stages.{i}.layers.{j}.'
                        f'pwconv1.bias'])}
                layers[f'{i}'][f'{j}']['pwconv2'] = {
                    'weight': pt2jx(hf_params[
                        f'{prefix}.encoder.stages.{i}.layers.{j}.'
                        f'pwconv2.weight'].T),
                    'bias': pt2jx(hf_params[
                        f'{prefix}.encoder.stages.{i}.layers.{j}.'
                        f'pwconv2.bias'])}
                if jx_config.grn_eps is not None:
                    layers[f'{i}'][f'{j}']['grn'] = {
                        'weight': pt2jx(hf_params[
                            f'{prefix}.encoder.stages.{i}.layers.{j}.'
                            f'grn.weight'])[0, 0, 0, :],
                        'bias': pt2jx(hf_params[
                            f'{prefix}.encoder.stages.{i}.layers.{j}.'
                            f'grn.bias'])[0, 0, 0, :]}
                else:
                    layers[f'{i}'][f'{j}']['layerscale'] = {
                        'weight': pt2jx(hf_params[
                            f'{prefix}.encoder.stages.{i}.layers.{j}.'
                            f'layer_scale_parameter'])}

        post_layernorm = {
            'weight': pt2jx(hf_params[f'{prefix}.layernorm.weight']),
            'bias': pt2jx(hf_params[f'{prefix}.layernorm.bias'])}
        head = {
            'weight': pt2jx(hf_params['classifier.weight']).T,
            'bias': pt2jx(hf_params['classifier.bias'])}

        params = {
            'embeddings': embeddings,
            'pre_layernorm': pre_layernorm,
            'layers': layers,
            'post_layernorm': post_layernorm}

        if jx_config.post_layernorm:
            params['post_layernorm'] = post_layernorm

        if jx_config.num_labels:
            params['head'] = head

        return params
