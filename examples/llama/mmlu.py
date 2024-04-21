"""Evaluating MMLU tasks."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import warnings
from argparse import ArgumentParser

import datasets
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import transformers
from einshard import einshard
from jax_smi import initialise_tracking
from scipy.optimize import minimize
from transformers import AutoTokenizer
from tqdm import tqdm
initialise_tracking()

from examples.default import get_args, str2bool
from transformerx.experimental.quantization import \
    AsymmetricQuantizedArray, \
    PowerSymmetricQuantizedArray, \
    SymmetricQuantizedArray
from transformerx.models.llama.default import \
    load_jx_config, load_jx_params, get_tokenize_fn
from transformerx.models.llama.modeling import forward_fn, LlamaInputs
from transformerx.tasks import \
    ARCEasy, ARCChallenge, CommonsenseQA, HellaSwag, PIQA
from transformerx.tasks.hendrycks_test import HendrycksTest, CATEGORIES


if __name__ == '__main__':

    # ----------------------------------------------------------------------- #
    # Command line arguments
    # ----------------------------------------------------------------------- #
    parser = ArgumentParser()

    parser.add_argument(
        '--model_name', default='huggyllama/llama-7b', type=str,
        help='a model name (default: huggyllama/llama-7b)')

    parser.add_argument(
        '--task_name', default='hendrycks_test', type=str,
        help='a task name (default: hendrycks_test)')
    parser.add_argument(
        '--n_fewshot', default=5, type=int,
        help='the number of few-shot examples (default: 5)')
    parser.add_argument(
        '--maxlen', default=2048, type=int,
        help='the maximum number of tokens (default: 2048)')

    parser.add_argument(
        '--quantization', default=None, type=str, choices=[
            'Q3_0', 'Q4_0', 'Q5_0', 'Q6_0', 'Q7_0', 'Q8_0',
            'PQ3_0', 'PQ4_0', 'PQ5_0', 'PQ6_0', 'PQ7_0', 'PQ8_0',
            'Q3_1', 'Q4_1', 'Q5_1', 'Q6_1', 'Q7_1', 'Q8_1',
        ], help='apply fake quantization if specified (default: None)')
    parser.add_argument(
        '--exponent', default=None, type=float,
        help='use global exponent parameter for PowerQuant (default: None)')

    args, print_fn = get_args(
        parser, exist_ok=True, dot_log_file=True,
        libraries=(datasets, jax, jaxlib, transformers))

    # ----------------------------------------------------------------------- #
    # Prepare task
    # ----------------------------------------------------------------------- #
    tasks = None # pylint: disable=invalid-name

    if args.task_name == 'arc_e':
        tasks = [ARCEasy(),]
        tasknames = ['arc_e',]

    if args.task_name == 'arc_c':
        tasks = [ARCChallenge(),]
        tasknames = ['arc_c',]

    if args.task_name == 'commonsense_qa':
        tasks = [CommonsenseQA(),]
        tasknames = ['commonsense_qa',]

    if args.task_name == 'hellaswag':
        tasks = [HellaSwag(),]
        tasknames = ['hellaswag',]

    if args.task_name == 'piqa':
        tasks = [PIQA(),]
        tasknames = ['piqa',]

    if args.task_name == 'hendrycks_test':
        tasks = [
            HendrycksTest(subject=s)
            for c in CATEGORIES for s in CATEGORIES[c]]
        tasknames = [s for c in CATEGORIES for s in CATEGORIES[c]]

    if tasks is None:
        raise NotImplementedError(f'Unknown args.task_name={args.task_name}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = sys.maxsize
    tokenizer.pad_token = tokenizer.eos_token

    maxlen = 0 # pylint: disable=invalid-name
    for task in tasks:
        example_docs = []
        if args.n_fewshot > 0:
            example_docs = task.kshot_docs()[:args.n_fewshot]
        for doc in task.valid_docs():
            maxlen = max(maxlen, len(tokenizer(
                task.create_qa_prompt_choices_fewshot(
                    example_docs, doc) + chr(65 + doc['gold'])).input_ids))

    maxlen = min(args.maxlen, maxlen)
    tokenize_fn = get_tokenize_fn(
        args.model_name, max_length=maxlen, add_special_tokens=True,
        padding_side='left', return_tensors='np')

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

    if config is None:
        raise NotImplementedError(f'Unknown args.model_name={args.model_name}')

    lm_head = params['lm_head']['weight']
    del params['lm_head']

    # ----------------------------------------------------------------------- #
    # Setup model
    # ----------------------------------------------------------------------- #
    if args.quantization in [f'Q{e}_0' for e in range(3, 9)]:
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
    
    if args.quantization in [f'PQ{e}_0' for e in range(3, 9)]:
        BITS = int(args.quantization[2])
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
            if args.exponent:
                qaram = PowerSymmetricQuantizedArray.quantize(
                    param, bits=BITS, contraction_axis=0, group_size=1,
                    exponent=args.exponent)
            else:
                with warnings.catch_warnings(), \
                        jax.default_device(jax.devices('cpu')[0]):
                    warnings.simplefilter('ignore')
                    def obj(args):
                        return np.mean((PowerSymmetricQuantizedArray.quantize(
                            param, bits=BITS, contraction_axis=0, group_size=1,
                            exponent=args[0]).materialize() - param)**2)
                    qaram = PowerSymmetricQuantizedArray.quantize(
                        param, bits=BITS, contraction_axis=0, group_size=1,
                        exponent=minimize(
                        obj, x0=np.array([1.0]), method='nelder-mead').x[0])
            return qaram.materialize()
        params = jax.tree_util.tree_map_with_path(_quantizer, params)

    if args.quantization in [f'Q{e}_1' for e in range(3, 9)]:
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
    # Compute classification metrics
    # ----------------------------------------------------------------------- #
    def _eval_doc(_doc, _examples, _task):

        prompt = _task.create_qa_prompt_choices_fewshot(_examples, _doc)
        inputs = LlamaInputs(**tokenize_fn([prompt]))

        # TODO: does it make sense?
        # check validity and get target token indices
        wordlines = tokenize_fn([
            prompt + ' ' + chr(65 + i)
            for i in range(len(_doc['choices']))])['input_ids']
        token_indices = []
        for wordline in wordlines:
            assert wordline[-2] == inputs.input_ids[0][-1]
            token_indices.append(wordline[-1])

        logits = forward_fn(params, inputs, config).last_hidden_states
        logits = logits[0, -1, :] @ lm_head[:, token_indices]
        lprobs = jax.nn.log_softmax(logits)
        return np.array(lprobs)

    all_docs = []
    all_log_probs = []
    all_metrics = []
    for task, taskname in zip(tasks, tasknames):
        example_docs = []
        if args.n_fewshot > 0:
            example_docs = task.kshot_docs()[:args.n_fewshot]

        docs = []
        log_probs = []
        for doc in task.valid_docs():
            docs.append(doc)
            log_probs.append(_eval_doc(doc, example_docs, task))
        all_docs += docs
        all_log_probs += log_probs

        metrics = task.evaluate(docs, log_probs)
        all_metrics.append(metrics)

        log_str = f'{taskname} ({len(docs)} docs) ' + ', '.join(f'{k}: {v:.3e}'
            for k, v in metrics.items() if isinstance(v, float))
        print_fn(log_str)

    if len(tasks) > 1:

        metrics = {
            k: sum(map(lambda e: e[k], all_metrics)) / len(all_metrics)
            for k in all_metrics[0] if isinstance(all_metrics[0][k], float)}
        log_str = 'Total (macro-average) ' + ', '.join(f'{k}: {v:.3e}'
            for k, v in metrics.items() if isinstance(v, float))
        print_fn(log_str)

        metrics = task.evaluate(all_docs, all_log_probs)
        log_str = 'Total (micro-average) ' + ', '.join(f'{k}: {v:.3e}'
            for k, v in metrics.items() if isinstance(v, float))
        print_fn(log_str)
