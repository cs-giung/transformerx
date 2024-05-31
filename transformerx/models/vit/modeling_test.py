"""Testing ViT."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import numpy as np
import jax
import jax.numpy as jnp
import torch
from einshard import einshard
from jax_smi import initialise_tracking
from transformers import AutoConfig, ViTModel
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
initialise_tracking()

from transformerx.models.vit.default import load_jx_config, load_jx_params
from transformerx.models.vit.modeling import ViTInputs, forward_fn


if __name__ == '__main__':

    NAME = 'cs-giung/vit-base-patch16-imagenet21k'
    CONFIG = AutoConfig.from_pretrained(NAME)

    # converters
    # pylint: disable=unnecessary-lambda-assignment
    pt2np = lambda e: e.cpu().numpy()
    jx2np = lambda e: np.asarray(e).copy()
    jx2pt = lambda e: torch.from_numpy(jx2np(e))

    @torch.no_grad
    def pt2jx(e): # pylint: disable=missing-function-docstring
        return jnp.asarray(e.cpu().numpy())

    input_pixels = 2.0 * (
        jax.random.uniform(jax.random.PRNGKey(0), (2, 224, 224, 3)) - 0.5)

    # transformers
    with torch.no_grad():
        model_pt = ViTModel.from_pretrained(NAME)
        inputs_pt = jx2pt(input_pixels).permute(0, 3, 1, 2)
        output_pt = model_pt(inputs_pt).pooler_output
        print(output_pt)

    # transformerx
    device = jax.devices('cpu')[0]
    with jax.default_device(device):
        params_jx = load_jx_params(NAME)
        del params_jx['head']
        config_jx = load_jx_config(NAME)
        inputs_jx = ViTInputs(input_pixels=input_pixels)
        output_jx = forward_fn(params_jx, inputs_jx, config_jx).pre_logits
        print(output_jx)
        abserr = np.abs(pt2np(output_pt) - jx2np(output_jx))
        print(f'- max: {abserr.max()}')
        print(f'- min: {abserr.min()}')

    # model parallel via einshard
    params_jx = jax.tree_util.tree_map(
        lambda e: einshard(e, '... O -> ... O*'), params_jx)
    inputs_jx = ViTInputs(input_pixels=input_pixels)
    output_jx = forward_fn(params_jx, inputs_jx, config_jx).pre_logits
    print(output_jx)
    abserr = np.abs(pt2np(output_pt) - jx2np(output_jx))
    print(f'- max: {abserr.max()}')
    print(f'- min: {abserr.min()}')
