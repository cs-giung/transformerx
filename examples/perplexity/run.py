"""Computing perplexity."""
import sys
from argparse import ArgumentParser
from functools import partial
sys.path.append('./') # pylint: disable=wrong-import-position

import datasets
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import transformers
from einshard import einshard
from jax_smi import initialise_tracking
from tqdm import tqdm
from transformers import AutoTokenizer
initialise_tracking()

from examples.default import get_args
from transformerx.experimental.quantization import SymmetricQuantizedArray
from transformerx.models.llama import default, rope
from transformerx.models.llama.modeling import forward_fn
from transformerx.models.llama.modeling import LlamaInputs as Inputs


if __name__ == '__main__':

    # ----------------------------------------------------------------------- #
    # Command line arguments
    # ----------------------------------------------------------------------- #
    parser = ArgumentParser()

    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--rope_type', required=True, type=str)

    parser.add_argument(
        '--data', default='wikitext2', type=str,
        help='data name (default: wikitext2)')
    parser.add_argument(
        '--seqlen', default=2048, type=int,
        help='a sequence length (default: 2048)')

    parser.add_argument(
        '--bits', default=0, type=int,
        help='apply fake quantization if specified (default: 0)')
    parser.add_argument(
        '--group_size', default=128, type=int,
        help='a group size to use for quantization (default: 128)')

    args, print_fn = get_args(
        parser, exist_ok=False, dot_log_file=False,
        libraries=(datasets, jax, jaxlib, transformers))

    # ----------------------------------------------------------------------- #
    # Prepare models
    # ----------------------------------------------------------------------- #
    config = default.load_jx_config(args.model)
    params = default.load_jx_params(args.model)

    # apply quantization
    if args.bits > 0:
        def _quantizer(path, param):
            if param.ndim < 2:
                return param
            if any(isinstance(e1, jax.tree_util.DictKey)
                and e1.key == 'lm_head' for e1 in path):
                return param
            if any(isinstance(e1, jax.tree_util.DictKey)
                and e1.key == 'embed_tokens' for e1 in path):
                return SymmetricQuantizedArray.quantize(
                    param, bits=args.bits,
                    contraction_axis=1, group_size=1).materialize()
            return SymmetricQuantizedArray.quantize(
                param, bits=args.bits,
                contraction_axis=0, group_size=args.group_size).materialize()
        params = jax.tree_util.tree_map_with_path(_quantizer, params)

    # ----------------------------------------------------------------------- #
    # Prepare datasets
    # ----------------------------------------------------------------------- #
    if args.data == 'wikitext2':
        tokens = datasets.load_dataset(
            'wikitext', 'wikitext-2-raw-v1', split='test')
        tokens = '\n\n'.join(tokens['text']) # pylint: disable=invalid-name

    elif args.data == 'ptb':
        tokens = datasets.load_dataset(
            'ptb_text_only', 'penn_treebank', split='test')
        tokens = ' '.join(tokens['sentence']) # pylint: disable=invalid-name

    else:
        raise NotImplementedError(
            f'Unknown args.data={args.data}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = sys.maxsize
    tokens = tokenizer(tokens, return_tensors='np').input_ids.reshape(-1)
    tokens = tokens[:tokens.size // args.seqlen * args.seqlen]
    tokens = tokens.reshape(-1, args.seqlen)

    # ----------------------------------------------------------------------- #
    # Compute perplexity
    # ----------------------------------------------------------------------- #
    params = jax.tree_util.tree_map(
        lambda e: einshard(e, '... O -> ... O*'), params)
    attention_mask = jnp.ones((1, args.seqlen))
    position_ids = jnp.arange(args.seqlen)[None, :]

    if args.rope_type == 'simple':
        make_rope = partial(
            rope.make_simple_rope,
            dim=config.hidden_size//config.num_attention_heads,
            base=config.rope_base)
    if args.rope_type == 'llama3':
        make_rope = partial(
            rope.make_llama3_rope,
            dim=config.hidden_size//config.num_attention_heads,
            base=config.rope_base)
    rope_cos, rope_sin = make_rope(position_ids)

    nlls = []
    ppls = []
    pbar = tqdm(tokens)
    for input_ids in pbar:
        input_ids[0] = tokenizer.bos_token_id
        inputs = Inputs(
            input_ids=input_ids[None],
            attention_mask=attention_mask,
            position_ids=position_ids,
            rope_cos=rope_cos,
            rope_sin=rope_sin)
        logits = forward_fn(params, inputs, config).logits
        lprobs = jnp.take_along_axis(
            jax.nn.log_softmax(logits[0, :-1, :]), input_ids[1:, None], axis=1)
        nlls.append(float(jnp.sum(jnp.negative(lprobs))))
        ppls.append(np.exp(sum(nlls) / len(nlls) / args.seqlen))
        pbar.set_description(f'PPL: {ppls[-1]:.3e}')
    print_fn('nlls=[' + ', '.join([f'{e:.3e}' for e in nlls]) + ']')
    print_fn('ppls=[' + ', '.join([f'{e:.3e}' for e in ppls]) + ']')
    print_fn(f'PPL: {ppls[-1]:.3e}')
    pbar.close()
