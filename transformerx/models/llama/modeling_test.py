# pylint: disable=duplicate-code
"""Testing Llama Model."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import numpy as np
import jax
import jax.numpy as jnp
import torch
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

from jax_smi import initialise_tracking
initialise_tracking()

from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
from transformerx.models.llama.default import \
    PREDEFINED_CONFIGS, convert_hf_params_to_jx_params
from transformerx.models.llama.modeling import LlamaInputs, forward_fn
from transformerx.utils.einshard import einshard


if __name__ == '__main__':

    NAME = 'meta-llama/Llama-2-7b-hf'
    CONFIG = AutoConfig.from_pretrained(NAME)

    tokenizer = AutoTokenizer.from_pretrained(NAME)
    tokenizer.pad_token = tokenizer.eos_token
    prompt = "Hey, are you conscious?" # pylint: disable=invalid-name

    # converters
    # pylint: disable=unnecessary-lambda-assignment
    pt2np = lambda e: e.cpu().numpy()
    jx2np = lambda e: np.asarray(e).copy()

    @torch.no_grad
    def pt2jx(e): # pylint: disable=missing-function-docstring
        return jnp.asarray(e.cpu().numpy())

    # transformers
    model_pt = LlamaForCausalLM.from_pretrained(NAME)
    params_pt = model_pt.state_dict()
    weight_pt = params_pt['lm_head.weight'].T
    with torch.no_grad():
        inputs_pt = tokenizer(prompt, return_tensors="pt")
        output_pt = model_pt(**inputs_pt, output_hidden_states=True)
        output_pt_hidden, output_pt_logits \
            = output_pt.hidden_states[-1], output_pt.logits

    # transformerx
    config_jx = PREDEFINED_CONFIGS[NAME]

    device = jax.devices('cpu')[0]
    with jax.default_device(device):
        params_jx = convert_hf_params_to_jx_params(params_pt)
        weight_jx = pt2jx(weight_pt)
        inputs_jx = tokenizer(prompt, return_tensors="jax")
        inputs_jx = LlamaInputs(
            input_ids=inputs_jx.input_ids,
            attention_mask=inputs_jx.attention_mask,
            position_ids=None)
        output_jx = forward_fn(params_jx, inputs_jx, config_jx)
        output_jx_hidden, output_jx_logits \
            = output_jx.last_hidden_states, output_jx.logits
        abserr = np.abs(pt2np(output_pt_hidden) - jx2np(output_jx_hidden))
        print(abserr.min(), abserr.max())
        abserr = np.abs(pt2np(output_pt_logits) - jx2np(output_jx_logits))
        print(abserr.min(), abserr.max())

    # model parallel via einshard
    params_jx = jax.tree_util.tree_map(
        lambda e: einshard(e, '... O -> ... O1'), params_jx)
    weight_jx = jax.tree_util.tree_map(
        lambda e: einshard(e, '... O -> ... O1'), weight_jx)

    inputs_jx = tokenizer(prompt, return_tensors="jax")
    inputs_jx = LlamaInputs(
        input_ids=inputs_jx.input_ids,
        attention_mask=inputs_jx.attention_mask,
        position_ids=None)
    output_jx = forward_fn(params_jx, inputs_jx, config_jx)
    output_jx_hidden, output_jx_logits \
        = output_jx.last_hidden_states, output_jx.logits
    abserr = np.abs(pt2np(output_pt_hidden) - jx2np(output_jx_hidden))
    print(abserr.min(), abserr.max())
    abserr = np.abs(pt2np(output_pt_logits) - jx2np(output_jx_logits))
    print(abserr.min(), abserr.max())
