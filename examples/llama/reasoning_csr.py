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
from transformerx.models.llama import default, rope
from transformerx.models.llama.modeling import forward_fn
from transformerx.models.llama.modeling import LlamaInputs as Inputs
from transformerx.tasks import \
    ARCEasy, ARCChallenge, CommonsenseQA, HellaSwag, PIQA


if __name__ == '__main__':

    # ----------------------------------------------------------------------- #
    # Command line arguments
    # ----------------------------------------------------------------------- #
    parser = ArgumentParser()

    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--rope_type', required=True, type=str)
    parser.add_argument('--tasks', required=True, type=lambda x: x.split(','))

    parser.add_argument(
        '--shot', default=0, type=int,
        help='the number of few-shot examples (default: 0)')
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
    config = default.load_jx_config(args.model)
    params = default.load_jx_params(args.model)
    tokenize_fn = default.get_tokenize_fn(
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
    # Prepare tasks
    # ----------------------------------------------------------------------- #
    tasks = []
    tasknames = args.tasks
    for taskname in tasknames:
        task = None
        if taskname == 'arc_e':
            task = ARCEasy()
        if taskname == 'arc_c':
            task = ARCChallenge()
        if taskname == 'commonsense_qa':
            task = CommonsenseQA()
        if taskname == 'hellaswag':
            task = HellaSwag()
        if taskname == 'piqa':
            task = PIQA()
        if task is None:
            raise NotImplementedError(f'Unknown taskname={taskname}')
        tasks.append(task)

    # ----------------------------------------------------------------------- #
    # Compute accuracy
    # ----------------------------------------------------------------------- #
    params = jax.tree_util.tree_map(
        lambda e: einshard(e, '... O -> ... O*'), params)

    if args.rope_type == 'simple':
        make_rope = partial(
            rope.make_simple_rope, dim=config.head_dim, base=config.rope_base)
    if args.rope_type == 'llama3':
        make_rope = partial(
            rope.make_llama3_rope, dim=config.head_dim, base=config.rope_base)

    data = []
    for taskname, task in zip(tasknames, tasks):
        kdocs = task.kshot_docs()[:args.shot] if args.shot > 0 else []
        correct = 0
        num_doc = 0
        for doc in task.valid_docs():
            prompt = task.create_qa_prompt_choices_fewshot(kdocs, doc)
            inputs = tokenize_fn([prompt])
            rope_cos, rope_sin = make_rope(inputs['position_ids'])
            inputs = Inputs(**inputs, rope_cos=rope_cos, rope_sin=rope_sin)
            answer = detokenize_fn([jnp.argmax(
                forward_fn(params, inputs, config).logits[0, -1, :])])
            correct += int(answer.strip() == chr(65 + doc['gold']))
            num_doc += 1
        data.append((taskname, correct, num_doc))
        print_fn(f'{taskname}: {correct} / {num_doc}')

    df = pd.DataFrame(
        data, columns=('taskname', 'correct', 'num_doc'))
    print_fn(f'Detailed results:\n{df}\n')

    for taskname in tasknames:
        c = df['taskname'] == taskname
        macro_average = (df[c]['correct'] / df[c]['num_doc']).mean()
        micro_average = df[c]['correct'].sum() / df[c]['num_doc'].sum()
        print_fn(f'{taskname} macro-average: {macro_average:.4f}')
        print_fn(f'{taskname} micro-average: {micro_average:.4f}')

    macro_average = (df['correct'] / df['num_doc']).mean()
    micro_average = df['correct'].sum() / df['num_doc'].sum()
    print_fn(f'total macro-average: {macro_average:.4f}')
    print_fn(f'total micro-average: {micro_average:.4f}')
