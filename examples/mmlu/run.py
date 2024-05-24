"""Run MMLU evaluation."""
import sys
from argparse import ArgumentParser
from functools import partial
sys.path.append('./') # pylint: disable=wrong-import-position

import datasets
import jax
import jax.numpy as jnp
import jaxlib
import pandas as pd
import transformers
from einshard import einshard
from jax_smi import initialise_tracking
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoTokenizer
initialise_tracking()

from examples.default import get_args
from transformerx.experimental.quantization import SymmetricQuantizedArray
from transformerx.tasks.hendrycks_test import HendrycksTest, CATEGORIES


if __name__ == '__main__':

    # ----------------------------------------------------------------------- #
    # Command line arguments
    # ----------------------------------------------------------------------- #
    parser = ArgumentParser()

    parser.add_argument('--model', required=True, type=str)

    parser.add_argument(
        '--shot', default=5, type=int,
        help='the number of few-shot examples (default: 5)')
    parser.add_argument(
        '--maxlen', default=2048, type=int,
        help='a maximum sequence length (default: 2048)')

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
            'meta-llama/Llama-2-7b-chat-hf',
            'meta-llama/Llama-2-13b-hf',
            'meta-llama/Llama-2-13b-chat-hf',
            'meta-llama/Llama-2-70b-hf',
            'meta-llama/Llama-2-70b-chat-hf',
            'meta-llama/Meta-Llama-3-8B',
            'meta-llama/Meta-Llama-3-8B-Instruct',
            'meta-llama/Meta-Llama-3-70B',
            'meta-llama/Meta-Llama-3-70B-Instruct',
            'mistralai/Mistral-7B-Instruct-v0.1',
            'mistralai/Mistral-7B-Instruct-v0.2',
            'mistralai/Mistral-7B-Instruct-v0.3',
            'microsoft/Phi-3-mini-4k-instruct',
            'microsoft/Phi-3-medium-4k-instruct',
        ):
        from transformerx.models.llama.default import \
            load_jx_config, load_jx_params, get_tokenize_fn
        from transformerx.models.llama.modeling import \
            forward_fn, LlamaInputs as Inputs

    else:
        raise NotImplementedError(
            f'Unknown args.model={args.model}')

    config = load_jx_config(args.model)
    params = load_jx_params(args.model)
    tokenize_fn = get_tokenize_fn(
        args.model, max_length=args.maxlen, add_special_tokens=True,
        padding_side='left', return_tensors='np')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    detokenize_fn = partial(tokenizer.decode, skip_special_tokens=True)

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
    # Compute accuracy
    # ----------------------------------------------------------------------- #
    params = jax.tree_util.tree_map(
        lambda e: einshard(e, '... O -> ... O*'), params)

    data = []
    for category in CATEGORIES:
        for subject in CATEGORIES[category]:
            task = HendrycksTest(subject=subject)
            kdocs = task.kshot_docs()[:args.shot]
            correct = 0
            num_doc = 0
            for doc in task.valid_docs():
                prompt = task.create_qa_prompt_choices_fewshot(kdocs, doc)
                inputs = Inputs(**tokenize_fn([prompt]))
                answer = detokenize_fn([jnp.argmax(
                    forward_fn(params, inputs, config).logits[0, -1, :])])
                correct += int(answer == chr(65 + doc['gold']))
                num_doc += 1
            data.append((category, subject, correct, num_doc))
            print_fn(f'{category}/{subject}: {correct} / {num_doc}')

    df = pd.DataFrame(
        data, columns=('category', 'subject', 'correct', 'num_doc'))
    print_fn(f'Detailed results:\n{df}\n')

    for category in CATEGORIES:
        c = df['category'] == category
        macro_average = (df[c]['correct'] / df[c]['num_doc']).mean()
        micro_average = df[c]['correct'].sum() / df[c]['num_doc'].sum()
        print_fn(f'{category} macro-average: {macro_average:.4f}')
        print_fn(f'{category} micro-average: {micro_average:.4f}')

    macro_average = (df['correct'] / df['num_doc']).mean()
    micro_average = df['correct'].sum() / df['num_doc'].sum()
    print_fn(f'total macro-average: {macro_average:.4f}')
    print_fn(f'total micro-average: {micro_average:.4f}')
