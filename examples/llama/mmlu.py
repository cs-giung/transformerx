"""Evaluating MMLU tasks."""
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
from transformerx.models.llama.default import \
    load_jx_config, load_jx_params, get_tokenize_fn
from transformerx.models.llama.modeling import forward_fn, LlamaInputs
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
        '--task_name', default='humanities', type=str,
        help='a task name (default: humanities)')

    parser.add_argument(
        '--quantization', default=None, type=str, choices=[
            'Q3_0', 'Q4_0', 'Q5_0', 'Q6_0', 'Q7_0', 'Q8_0',
            'Q3_1', 'Q4_1', 'Q5_1', 'Q6_1', 'Q7_1', 'Q8_1',
        ], help='apply fake quantization if specified (default: None)')

    args, print_fn = get_args(
        parser, exist_ok=True, dot_log_file=True,
        libraries=(datasets, jax, jaxlib, transformers))

    # ----------------------------------------------------------------------- #
    # Prepare task
    # ----------------------------------------------------------------------- #
    tasks = [HendrycksTest(subject=e) for e in CATEGORIES[args.task_name]]

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
        ):
        config = load_jx_config(args.model_name)
        params = load_jx_params(args.model_name)

    if config is None:
        raise NotImplementedError(f'Unknown args.model_name={args.model_name}')

    maxlen = 2048
    tokenize_fn = get_tokenize_fn(
        args.model_name, max_length=maxlen, add_special_tokens=False,
        padding_side='left', return_tensors='np')

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
                param, bits=BITS,
                contraction_axis=0, group_size=1).materialize()
            return qaram
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
                param, bits=BITS,
                contraction_axis=0, group_size=1).materialize()
            return qaram
        params = jax.tree_util.tree_map_with_path(_quantizer, params)

    params = jax.tree_util.tree_map(
        lambda e: einshard(e, '... O -> ... O*'), params)

    # ----------------------------------------------------------------------- #
    # Compute classification metrics
    # ----------------------------------------------------------------------- #
    def _eval_doc(_doc, _examples, _task):

        prompt = _task.create_fewshot_prompt(_examples, _doc)
        sentence = tokenize_fn([prompt + e for e in _doc['choices']])
        question = tokenize_fn([prompt     for e in _doc['choices']])

        inputs = LlamaInputs(**sentence)
        logits = forward_fn(params, inputs, config).logits
        _score = []
        for idx in range(logits.shape[0]):
            ans = sentence['input_ids'][idx, -1:]
            gss = jax.nn.log_softmax(logits[idx, -2:-1])
            _score.append(np.array(
                jnp.take_along_axis(gss, ans[:, None], axis=-1)))
        return _score

    log_probs_u = []
    log_probs_t = []
    log_probs_b = []
    docs = []
    for task in tasks:
        for doc in tqdm(task.valid_docs()):
            examples = task.kshot_docs()[:5]
            _score = _eval_doc(doc, examples, task)
            docs.append(doc)

            # unnormalized
            log_prob = np.array([e.sum() for e in _score])
            log_prob = np.exp(log_prob - log_prob.max())
            log_prob = np.log(log_prob / log_prob.sum())
            log_probs_u.append(log_prob)

            # token-length normalized
            log_prob = np.array([e.sum() for e in _score])
            log_prob = log_prob / np.array([len(e) for e in _score])
            log_prob = np.exp(log_prob - log_prob.max())
            log_prob = np.log(log_prob / log_prob.sum())
            log_probs_t.append(log_prob)

            # byte-length normalized
            log_prob = np.array([e.sum() for e in _score])
            log_prob = log_prob / np.array([len(e) for e in doc['choices']])
            log_prob = np.exp(log_prob - log_prob.max())
            log_prob = np.log(log_prob / log_prob.sum())
            log_probs_b.append(log_prob)

    metrics = task.evaluate(docs, log_probs_u)
    log_str = 'Unnormalized:' + ', '.join(f'{k}: {v:.3e}'
        for k, v in metrics.items() if isinstance(v, float))
    print_fn(log_str)

    metrics = task.evaluate(docs, log_probs_t)
    log_str = 'Token-length normalized:' + ', '.join(f'{k}: {v:.3e}'
        for k, v in metrics.items() if isinstance(v, float))
    print_fn(log_str)

    metrics = task.evaluate(docs, log_probs_b)
    log_str = 'Byte-length normalized:' + ', '.join(f'{k}: {v:.3e}'
        for k, v in metrics.items() if isinstance(v, float))
    print_fn(log_str)
