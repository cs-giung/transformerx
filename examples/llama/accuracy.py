"""Evaluating zero-shot tasks."""
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
from transformerx.tasks import ARCEasy, ARCChallenge, HellaSwag, PIQA


if __name__ == '__main__':

    # ----------------------------------------------------------------------- #
    # Command line arguments
    # ----------------------------------------------------------------------- #
    parser = ArgumentParser()

    parser.add_argument(
        '--model_name', default='huggyllama/llama-7b', type=str,
        help='a model name (default: huggyllama/llama-7b)')

    parser.add_argument(
        '--task_name', default='arc_e', type=str,
        help='a task name (default: arc_e)')

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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = sys.maxsize
    tokenizer.pad_token = tokenizer.eos_token

    task = None # pylint: disable=invalid-name
    if args.task_name == 'arc_e':
        task = ARCEasy()
    if args.task_name == 'arc_c':
        task = ARCChallenge()
    if args.task_name == 'hellaswag':
        task = HellaSwag()
    if args.task_name == 'piqa':
        task = PIQA()
    if task is None:
        raise NotImplementedError(f'Unknown args.task_name={args.task_name}')

    maxlen = 0 # pylint: disable=invalid-name
    for doc in task.valid_docs():
        maxlen = max(maxlen, max(len(e) for e in tokenizer([
            doc['query'] + e for e in doc['choices']]).input_ids))
    tokenize_fn = get_tokenize_fn(
        args.model_name, max_length=maxlen, add_special_tokens=False,
        padding_side='right', return_tensors='np')

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
    def _eval_doc(_doc):
        sentence = tokenize_fn([_doc['query'] + e for e in _doc['choices']])
        question = tokenize_fn([_doc['query']     for e in _doc['choices']])

        inputs = LlamaInputs(**sentence)
        logits = forward_fn(params, inputs, config).logits
        lprobs = []
        for idx in range(logits.shape[0]):
            ans = sentence['input_ids'][
                idx, jnp.sum(question['attention_mask'][idx]):
                     jnp.sum(sentence['attention_mask'][idx])]
            gss = jax.nn.log_softmax(logits[
                idx, jnp.sum(question['attention_mask'][idx]) - 1:
                     jnp.sum(sentence['attention_mask'][idx]) - 1])
            lprobs.append(float(
                jnp.sum(jnp.take_along_axis(gss, ans[:, None], axis=-1))))

        cat = np.array(lprobs)
        cat = np.exp(cat - cat.max())
        cat = cat / cat.sum()
        nll = np.negative(np.log(cat[doc['gold']]))
        acc = np.argmax(lprobs) == doc['gold']

        return acc, nll

    accs = []
    nlls = []
    with tqdm(task.valid_docs()) as pbar:
        for doc in pbar:
            acc, nll = _eval_doc(doc)
            accs.append(acc)
            nlls.append(nll)
            acc = np.mean(accs)
            nll = np.mean(nlls)
            pbar.set_description(f'ACC: {acc:.3e}, NLL: {nll:.3e}')
        print_fn(f'ACC: {acc:.3e}, NLL: {nll:.3e}')
