"""Testing Phi."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import numpy as np
import jax
import jax.numpy as jnp
import torch
from einshard import einshard
from jax_smi import initialise_tracking
from transformers import AutoConfig, AutoTokenizer, Phi3ForCausalLM
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
initialise_tracking()

from transformerx.models.phi.default import \
    load_jx_config, load_jx_params, get_tokenize_fn
from transformerx.models.phi.modeling import PhiInputs, forward_fn


if __name__ == '__main__':

    NAME = 'microsoft/Phi-3-mini-4k-instruct'
    CONFIG = AutoConfig.from_pretrained(NAME)
    PROMPT = [
        "Hey, are you conscious? This is",
        "I want to verify the batched inference with some"]
    MAX_LENGTH = 32

    # converters
    # pylint: disable=unnecessary-lambda-assignment
    pt2np = lambda e: e.cpu().numpy()
    jx2np = lambda e: np.asarray(e).copy()

    @torch.no_grad
    def pt2jx(e): # pylint: disable=missing-function-docstring
        return jnp.asarray(e.cpu().numpy())

    # transformers
    with torch.no_grad():
        model_pt = Phi3ForCausalLM.from_pretrained(NAME)
        model_pt.eval()

        tokenizer = AutoTokenizer.from_pretrained(NAME)
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token
        inputs_pt = tokenizer(
            PROMPT, return_tensors='pt',
            padding='max_length', max_length=MAX_LENGTH)
        output_pt = model_pt(**inputs_pt, output_hidden_states=True)
        output_pt_hidden, output_pt_logits \
            = output_pt.hidden_states[-1], output_pt.logits
        print(tokenizer.batch_decode(output_pt_logits.argmax(-1)))

        tokenizer = AutoTokenizer.from_pretrained(NAME)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        inputs_pt = tokenizer(
            PROMPT, return_tensors='pt',
            padding='max_length', max_length=MAX_LENGTH)
        output_pt = model_pt(**inputs_pt, output_hidden_states=True)
        output_pt_hidden, output_pt_logits \
            = output_pt.hidden_states[-1], output_pt.logits
        print(tokenizer.batch_decode(output_pt_logits.argmax(-1)))

    # transformerx
    config_jx = load_jx_config(NAME)

    device = jax.devices('cpu')[0]
    with jax.default_device(device):
        params_jx = load_jx_params(NAME)

        tokenize = get_tokenize_fn(
            NAME, max_length=MAX_LENGTH,
            padding_side='right', return_tensors='jax')
        inputs_jx = tokenize(PROMPT)
        inputs_jx = PhiInputs(
            input_ids=inputs_jx['input_ids'],
            attention_mask=inputs_jx['attention_mask'],
            position_ids=inputs_jx['position_ids'])
        output_jx = forward_fn(params_jx, inputs_jx, config_jx)
        output_jx_hidden, output_jx_logits \
            = output_jx.last_hidden_states, output_jx.logits
        print(tokenizer.batch_decode(output_jx_logits.argmax(-1)))

        tokenize = get_tokenize_fn(
            NAME, max_length=MAX_LENGTH,
            padding_side='left', return_tensors='jax')
        inputs_jx = tokenize(PROMPT)
        inputs_jx = PhiInputs(
            input_ids=inputs_jx['input_ids'],
            attention_mask=inputs_jx['attention_mask'],
            position_ids=inputs_jx['position_ids'])
        output_jx = forward_fn(params_jx, inputs_jx, config_jx)
        output_jx_hidden, output_jx_logits \
            = output_jx.last_hidden_states, output_jx.logits
        print(tokenizer.batch_decode(output_jx_logits.argmax(-1)))

    # model parallel via einshard
    params_jx = jax.tree_util.tree_map(
        lambda e: einshard(e, '... O -> ... O*'), params_jx)

    tokenize = get_tokenize_fn(
        NAME, max_length=MAX_LENGTH,
        padding_side='right', return_tensors='jax')
    inputs_jx = tokenize(PROMPT)
    inputs_jx = PhiInputs(
        input_ids=inputs_jx['input_ids'],
        attention_mask=inputs_jx['attention_mask'],
        position_ids=inputs_jx['position_ids'])
    output_jx = forward_fn(params_jx, inputs_jx, config_jx)
    output_jx_hidden, output_jx_logits \
        = output_jx.last_hidden_states, output_jx.logits
    print(tokenizer.batch_decode(output_jx_logits.argmax(-1)))

    tokenize = get_tokenize_fn(
        NAME, max_length=MAX_LENGTH,
        padding_side='left', return_tensors='jax')
    inputs_jx = tokenize(PROMPT)
    inputs_jx = PhiInputs(
        input_ids=inputs_jx['input_ids'],
        attention_mask=inputs_jx['attention_mask'],
        position_ids=inputs_jx['position_ids'])
    output_jx = forward_fn(params_jx, inputs_jx, config_jx)
    output_jx_hidden, output_jx_logits \
        = output_jx.last_hidden_states, output_jx.logits
    print(tokenizer.batch_decode(output_jx_logits.argmax(-1)))
