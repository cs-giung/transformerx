"""Calculating perplexity."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

from argparse import ArgumentParser

import datasets
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import transformers
from einshard import einshard
from jax_smi import initialise_tracking
from transformers import AutoTokenizer
from tqdm import tqdm
initialise_tracking()

from examples.default import get_args, str2bool
from transformerx.experimental.quantization import \
    AsymmetricQuantizedArray, SymmetricQuantizedArray
from transformerx.models.llama.default import load_jx_config, load_jx_params
from transformerx.models.llama.modeling import forward_fn, LlamaInputs


if __name__ == '__main__':

    # ----------------------------------------------------------------------- #
    # Command line arguments
    # ----------------------------------------------------------------------- #
    parser = ArgumentParser()

    parser.add_argument(
        '--model_name', default='huggyllama/llama-7b', type=str,
        help='a model name (default: huggyllama/llama-7b)')

    parser.add_argument(
        '--data_name', default='wikitext2', type=str,
        help='a data name (default: wikitext2)')
    parser.add_argument(
        '--ensure_bos', default=False, type=str2bool,
        help='the first token will be <BOS> if specified (default: False)')
    parser.add_argument(
        '--seqlen', default=2048, type=int,
        help='a sequence length for evaluating perplexity (default: 2048)')

    parser.add_argument(
        '--quantization', default=None, type=str, choices=[
            'Q3_0', 'Q4_0', 'Q5_0', 'Q6_0', 'Q7_0', 'Q8_0',
            'Q3_1', 'Q4_1', 'Q5_1', 'Q6_1', 'Q7_1', 'Q8_1',
        ], help='apply fake quantization if specified (default: None)')

    args, print_fn = get_args(
        parser, exist_ok=True, dot_log_file=True,
        libraries=(datasets, jax, jaxlib, transformers))

    # ----------------------------------------------------------------------- #
    # Preprocess dataset
    # ----------------------------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = sys.maxsize
    tokens = None # pylint: disable=invalid-name

    if args.data_name == 'wikitext2':
        tokens = datasets.load_dataset(
            'wikitext', 'wikitext-2-raw-v1', split='test')
        tokens = tokenizer(
            '\n\n'.join(tokens['text']), return_tensors='np').input_ids

    if args.data_name == 'ptb':
        tokens = datasets.load_dataset(
            'ptb_text_only', 'penn_treebank', split='test')
        tokens = tokenizer(
            ' '.join(tokens['sentence']), return_tensors='np').input_ids

    if tokens is None:
        raise NotImplementedError(f'Unknown args.data_name={args.data_name}')

    # ----------------------------------------------------------------------- #
    # Load model
    # ----------------------------------------------------------------------- #
    config = None # pylint: disable=invalid-name
    params = None # pylint: disable=invalid-name

    if args.model_name in (
            'huggyllama/llama-7b',
            'huggyllama/llama-13b',
            'huggyllama/llama-30b',
            'huggyllama/llama-65b',
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-13b-hf',
            'meta-llama/Llama-2-70b-hf',
            'meta-llama/Meta-Llama-3-8B',
            'meta-llama/Meta-Llama-3-70B',
        ):
        config = load_jx_config(args.model_name)
        params = load_jx_params(args.model_name)

        def packing_inputs(input_ids):
            # pylint: disable=missing-function-docstring,redefined-outer-name
            return LlamaInputs(
                input_ids=input_ids, attention_mask=jnp.ones_like(input_ids),
                position_ids=jnp.arange(input_ids.shape[1])[None, :])

    if config is None:
        raise NotImplementedError(f'Unknown args.model_name={args.model_name}')

    # ----------------------------------------------------------------------- #
    # Setup model
    # ----------------------------------------------------------------------- #
    if args.quantization in ['Q3_0', 'Q4_0', 'Q5_0', 'Q6_0', 'Q7_0', 'Q8_0']:
        BITS = int(args.quantization[1])
        SKIP_PATTERNS = ('embed_tokens', 'lm_head')
        def _quantizer(path, param):
            if param.ndim < 2:
                return param
            if any(isinstance(e1, jax.tree_util.GetAttrKey) and any(
                    e2 in e1.name for e2 in SKIP_PATTERNS) for e1 in path):
                return param
            if any(isinstance(e1, jax.tree_util.DictKey) and any(
                    e2 in e1.key for e2 in SKIP_PATTERNS) for e1 in path):
                return param
            qaram = SymmetricQuantizedArray.quantize(
                param, bits=BITS, contraction_axis=0, group_size=1)
            return qaram.materialize()
        params = jax.tree_util.tree_map_with_path(_quantizer, params)

    if args.quantization in ['Q3_1', 'Q4_1', 'Q5_1', 'Q6_1', 'Q7_1', 'Q8_1']:
        BITS = int(args.quantization[1])
        SKIP_PATTERNS = ('embed_tokens', 'lm_head')
        def _quantizer(path, param):
            if param.ndim < 2:
                return param
            if any(isinstance(e1, jax.tree_util.GetAttrKey) and any(
                    e2 in e1.name for e2 in SKIP_PATTERNS) for e1 in path):
                return param
            if any(isinstance(e1, jax.tree_util.DictKey) and any(
                    e2 in e1.key for e2 in SKIP_PATTERNS) for e1 in path):
                return param
            qaram = AsymmetricQuantizedArray.quantize(
                param, bits=BITS, contraction_axis=0, group_size=1)
            return qaram.materialize()
        params = jax.tree_util.tree_map_with_path(_quantizer, params)

    params = jax.tree_util.tree_map(
        lambda e: einshard(e, '... O -> ... O*'), params)

    # ----------------------------------------------------------------------- #
    # Compute perplexity
    # ----------------------------------------------------------------------- #
    seqlen = args.seqlen
    nsamples = tokens.size // seqlen

    nlls = []
    with tqdm(range(nsamples)) as pbar:
        for i in pbar:

            labels = tokens[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:][0]
            input_ids = tokens[:, (i * seqlen) : ((i + 1) * seqlen)]
            if args.ensure_bos:
                input_ids[0][0] = tokenizer.bos_token_id

            inputs = packing_inputs(input_ids)
            logits = forward_fn(params, inputs, config).logits
            logits = logits[:, :-1, :][0]
            lprobs = jax.nn.log_softmax(logits)
            lprobs = jnp.take_along_axis(lprobs, labels[:, None], axis=-1)
            nlls.append(float(jnp.sum(jnp.negative(lprobs))))

            ppl = np.exp(sum(nlls) / len(nlls) / seqlen)
            pbar.set_description(f'PPL: {ppl:.3e}')

        print_fn(f'PPL: {ppl:.3e}')
