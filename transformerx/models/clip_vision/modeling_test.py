"""Testing CLIP Vision Model."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import jax
import jax.numpy as jnp
import numpy as np
import torch

from transformers import AutoConfig,CLIPVisionModel
from transformerx.models.clip_vision.default import \
    load_jx_config, load_jx_params
from transformerx.models.clip_vision.modeling import \
    CLIPVisionInputs, forward_fn


if __name__ == '__main__':

    NAME = 'openai/clip-vit-large-patch14'
    CONFIG = AutoConfig.from_pretrained(NAME)

    # converters
    # pylint: disable=unnecessary-lambda-assignment
    pt2np = lambda e: e.cpu().numpy()
    jx2np = lambda e: np.asarray(e).copy()
    jx2pt = lambda e: torch.from_numpy(jx2np(e))

    @torch.no_grad
    def pt2jx(e): # pylint: disable=missing-function-docstring
        return jnp.asarray(e.cpu().numpy())

    input_pixels = jax.random.uniform(jax.random.PRNGKey(0), (2, 224, 224, 3))

    # transformers
    with torch.no_grad():
        model_pt = CLIPVisionModel.from_pretrained(NAME)
        inputs_pt = jx2pt(input_pixels).permute(0, 3, 1, 2)
        output_pt = model_pt(inputs_pt).pooler_output

    # transformerx
    device = jax.devices('cpu')[0]
    with jax.default_device(device):
        params_jx = load_jx_params(NAME)
        config_jx = load_jx_config(NAME)
        inputs_jx = CLIPVisionInputs(input_pixels=input_pixels)
        output_jx = forward_fn(
            params_jx, inputs_jx, config_jx).last_hidden_states
        abserr = np.abs(pt2np(output_pt) - jx2np(output_jx))
        print(f'torch - jax ({device})')
        print(f'- max: {abserr.max()}')
        print(f'- min: {abserr.min()}')
