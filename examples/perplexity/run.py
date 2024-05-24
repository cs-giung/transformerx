"""Computing perplexity."""
import sys
from argparse import ArgumentParser
sys.path.append('./') # pylint: disable=wrong-import-position

import datasets
import jax
import jax.numpy as jnp
import jaxlib
import transformers
from einshard import einshard
from jax_smi import initialise_tracking
from tqdm import tqdm
from transformers import AutoTokenizer
initialise_tracking()

from examples.default import get_args
from transformerx.experimental.quantization import SymmetricQuantizedArray


if __name__ == '__main__':

    # ----------------------------------------------------------------------- #
    # Command line arguments
    # ----------------------------------------------------------------------- #
    parser = ArgumentParser()

    parser.add_argument('--model', required=True, type=str)

    parser.add_argument(
        '--data', default='wikitext2', type=str,
        help='data name (default: wikitext2)')
    parser.add_argument(
        '--seqlen', default=2048, type=int,
        help='a sequence length (default: 2048)')

    parser.add_argument(
        '--quant', default=None, type=str, choices=[
            'Q3_0', 'Q4_0', 'Q5_0', 'Q6_0', 'Q7_0', 'Q8_0',
        ], help='apply fake quantization if specified (default: None)')

    args, print_fn = get_args(
        parser, exist_ok=True, dot_log_file=False,
        libraries=(datasets, jax, jaxlib, transformers))

    # ----------------------------------------------------------------------- #
    # Prepare models
    # ----------------------------------------------------------------------- #
    if args.model in (
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
        from transformerx.models.llama.default import \
            load_jx_config, load_jx_params
        from transformerx.models.llama.modeling import \
            forward_fn, LlamaInputs as Inputs

    elif args.model in (
            'microsoft/Phi-3-mini-4k-instruct',
        ):
        from transformerx.models.phi.default import \
            load_jx_config, load_jx_params
        from transformerx.models.phi.modeling import \
            forward_fn, PhiInputs as Inputs

    else:
        raise NotImplementedError(
            f'Unknown args.model={args.model}')

    config = load_jx_config(args.model)
    params = load_jx_params(args.model)

    # apply fake quantization
    if args.quant in [f'Q{i}_0' for i in range(3, 9)]:
        BITS = int(args.quant[1])
        SKIP_PATTERNS = ('lm_head',)
        def _quantizer(path, param):
            if param.ndim < 2:
                return param
            if any(isinstance(e1, jax.tree_util.DictKey)
                and e1.key == 'lm_head' for e1 in path):
                return param
            if any(isinstance(e1, jax.tree_util.DictKey)
                and e1.key == 'embed_tokens' for e1 in path):
                return SymmetricQuantizedArray.quantize(
                    param, bits=BITS, contraction_axis=1, group_size=1
                ).materialize()
            return SymmetricQuantizedArray.quantize(
                param, bits=BITS, contraction_axis=0, group_size=1
            ).materialize()
        params = jax.tree_util.tree_map_with_path(_quantizer, params)

    elif args.quant is not None:
        raise NotImplementedError(
            f'Unknown args.quant={args.quant}')

    # ----------------------------------------------------------------------- #
    # Prepare datasets
    # ----------------------------------------------------------------------- #
    if args.data == 'wikitext2':
        tokens = datasets.load_dataset(
            'wikitext', 'wikitext-2-raw-v1', split='test')
        tokens = '\n\n'.join(tokens['text']) # pylint: disable=invalid-name

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
    position_ids = jnp.ones((1, args.seqlen))

    nlls = []
    ppls = []
    pbar = tqdm(tokens)
    for input_ids in pbar:
        input_ids[0] = tokenizer.bos_token_id
        inputs = Inputs(
            input_ids=input_ids[None],
            attention_mask=attention_mask,
            position_ids=position_ids)
        logits = forward_fn(params, inputs, config).logits
        lprobs = jnp.take_along_axis(
            jax.nn.log_softmax(logits[0, :-1, :]), input_ids[1:, None], axis=1)
        nlls.append(float(jnp.sum(jnp.negative(lprobs))))
        ppls.append(sum(nlls) / len(nlls) / args.seqlen)
        pbar.set_description(f'PPL: {ppls[-1]:.3e}')
    print_fn('nlls=[' + ', '.join([f'{e:.3e}' for e in nlls]) + ']')
    print_fn('ppls=[' + ', '.join([f'{e:.3e}' for e in ppls]) + ']')
    print_fn(f'PPL: {ppls[-1]:.3e}')
    pbar.close()
