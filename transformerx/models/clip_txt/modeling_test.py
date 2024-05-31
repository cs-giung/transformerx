"""Testing CLIP-ViT."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import numpy as np
import jax
import jax.numpy as jnp
import torch
from einshard import einshard
from jax_smi import initialise_tracking
from transformers import AutoConfig, AutoTokenizer, CLIPTextModel
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
initialise_tracking()

from transformerx.models.clip_txt.default import \
    load_jx_config, load_jx_params, get_tokenize_fn
from transformerx.models.clip_txt.modeling import \
    CLIPTxTInputs, forward_fn


if __name__ == '__main__':

    NAME = 'openai/clip-vit-large-patch14'
    CONFIG = AutoConfig.from_pretrained(NAME)
    PROMPT = [
        "Hey, are you conscious? This is",
        "I want to verify the batched inference with some"]
    MAX_LENGTH = 32

    # converters
    # pylint: disable=unnecessary-lambda-assignment
    pt2np = lambda e: e.cpu().numpy()
    jx2np = lambda e: np.asarray(e).copy()
    jx2pt = lambda e: torch.from_numpy(jx2np(e))

    @torch.no_grad
    def pt2jx(e): # pylint: disable=missing-function-docstring
        return jnp.asarray(e.cpu().numpy())

    # transformers
    with torch.no_grad():
        model_pt = CLIPTextModel.from_pretrained(NAME)
        model_pt.eval()

        tokenizer = AutoTokenizer.from_pretrained(NAME)
        tokenizer.padding_side = 'right'
        inputs_pt = tokenizer(
            PROMPT, return_tensors='pt',
            padding='max_length', max_length=MAX_LENGTH)
        output_pt = model_pt(**inputs_pt).pooler_output

    # transformerx
    device = jax.devices('cpu')[0]
    with jax.default_device(device):
        params_jx = load_jx_params(NAME)
        config_jx = load_jx_config(NAME)

        tokenize = get_tokenize_fn(
            NAME, max_length=MAX_LENGTH,
            padding_side='right', return_tensors='jax')
        inputs_jx = tokenize(PROMPT)
        inputs_jx = CLIPTxTInputs(
            input_ids=inputs_jx['input_ids'],
            attention_mask=inputs_jx['attention_mask'],
            position_ids=inputs_jx['position_ids'])
        output_jx = forward_fn(
            params_jx, inputs_jx, config_jx).last_hidden_states
        abserr = np.abs(pt2np(output_pt) - jx2np(output_jx))
        print(f'- max: {abserr.max()}')
        print(f'- min: {abserr.min()}')

    # model parallel via einshard
    params_jx = jax.tree_util.tree_map(
        lambda e: einshard(e, '... O -> ... O*'), params_jx)
    output_jx = forward_fn(
        params_jx, inputs_jx, config_jx).last_hidden_states
    abserr = np.abs(pt2np(output_pt) - jx2np(output_jx))
    print(f'- max: {abserr.max()}')
    print(f'- min: {abserr.min()}')
