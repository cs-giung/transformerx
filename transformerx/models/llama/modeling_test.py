# pylint: disable=duplicate-code
"""Testing Llama Model."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import numpy as np
import jax
import jax.numpy as jnp
import torch

from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
from transformerx.models.llama.default import \
    PREDEFINED_CONFIGS, convert_hf_params_to_jx_params
from transformerx.models.llama.modeling import LlamaInputs, forward_fn


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
    with torch.no_grad():
        inputs_pt = tokenizer(prompt, return_tensors="pt")
        output_pt = model_pt(
            **inputs_pt, output_hidden_states=True).hidden_states[-1]
        output_pt_logits = output_pt @ params_pt['lm_head.weight'].T

    # transformerx
    params_jx = convert_hf_params_to_jx_params(params_pt)
    config_jx = PREDEFINED_CONFIGS[NAME]

    # TODO: when device is not CPU
    for device in [jax.devices('cpu')[0],]:
        with jax.default_device(device):
            inputs_jx = tokenizer(prompt, return_tensors="jax")
            inputs_jx = LlamaInputs(
                input_ids=inputs_jx.input_ids,
                attention_mask=inputs_jx.attention_mask,
                position_ids=jnp.arange(
                    0, inputs_jx.input_ids.shape[1]).astype(int)[None, :])
            output_jx = forward_fn(params_jx, inputs_jx, config_jx)

            abserr = np.abs(pt2np(output_pt) - jx2np(output_jx))
            print(abserr.min(), abserr.max())

            output_jx_logits = output_jx @ pt2jx(params_pt['lm_head.weight']).T
            print(output_pt_logits)
            print(output_jx_logits)
