"""Fine-tuning"""
import os
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

try:
    import wandb
except ImportError:
    import warnings
    warnings.warn('Failed to import wandb')

from argparse import ArgumentParser
from collections import OrderedDict
from functools import partial

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import tensorflow
import tensorflow_datasets
import transformers
from jax_smi import initialise_tracking
initialise_tracking()

from examples.default import get_args, str2bool
from examples.finetuning.input_pipeline import create_trn_iter, create_val_iter
from transformerx.experimental.optimization import adam, ivon


if __name__ == '__main__':

    # ----------------------------------------------------------------------- #
    # Command line arguments
    # ----------------------------------------------------------------------- #
    parser = ArgumentParser()

    parser.add_argument(
        '--model_name', default='openai/clip-vit-large-patch14', type=str,
        help='(default: openai/clip-vit-large-patch14)')

    parser.add_argument(
        '--data_name', default='imagenet2012', type=str,
        help='(default: imagenet2012)')

    parser.add_argument(
        '--optim', default='adam', type=str,
        help='optimization method (default: adam)')
    parser.add_argument(
        '--optim_batch_size', default=64, type=int,
        help='a size of mini-batch for each training step (default: 64)')
    parser.add_argument(
        '--optim_num_steps', default=5000, type=int,
        help='the number of training steps (default: 5000)')
    parser.add_argument(
        '--optim_learning_rate', default=1e-03, type=float,
        help='a peak learning rate for training (default: 1e-03)')
    parser.add_argument(
        '--optim_momentum_mu', default=0.9, type=float,
        help='a momentum coefficient (default: 0.9)')
    parser.add_argument(
        '--optim_momentum_nu', default=0.999, type=float,
        help='a momentum coefficient (default: 0.999)')
    parser.add_argument(
        '--optim_weight_decay', default=0.1, type=float,
        help='an weight decay coefficient (default: 0.1)')
    parser.add_argument(
        '--optim_clip_radius', default=None, type=float,
        help='clipping update if specified (default: None)')
    parser.add_argument(
        '--optim_ess_factor', default=1.0, type=float,
        help='effective sample size factor (default: 1.0)')

    parser.add_argument(
        '--wandb', default=False, type=str2bool,
        help='use wandb for logging if specified (default: False)')
    parser.add_argument(
        '--wandb_project', default='transformerx', type=str,
        help='a name of project receiving logs (default: transformerx)')
    parser.add_argument(
        '--wandb_entity', default='cs-giung', type=str,
        help='a name of user or team receiving logs (default: cs-giung)')

    args, print_fn = get_args(
        parser, exist_ok=False, dot_log_file=False,
        libraries=(jax, jaxlib, tensorflow, tensorflow_datasets, transformers))

    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        wandb.config.update(args)

    # ----------------------------------------------------------------------- #
    # Prepare dataset
    # ----------------------------------------------------------------------- #
    dataset_builder = tensorflow_datasets.builder(args.data_name)
    input_shape = (224, 224, 3)
    shard_shape = (jax.local_device_count(), -1)
    num_classes = {
        'domainnet': 345,
        'imagenet2012': 1000,
    }[args.data_name]

    trn_split = {
        'domainnet': 'train[:99%]',
        'imagenet2012': 'train[:99%]',
    }[args.data_name]
    trn_iter, trn_dataset_size, trn_steps_per_epoch = create_trn_iter(
        dataset_builder, args.optim_batch_size, shard_shape, split=trn_split)
    log_str = (
        f'It will go through {trn_steps_per_epoch} steps to handle '
        f'{trn_dataset_size} training data.')
    print_fn(log_str)

    val_split = {
        'domainnet': 'train[99%:]',
        'imagenet2012': 'train[99%:]',
    }[args.data_name]
    val_iter, val_dataset_size, val_steps_per_epoch = create_val_iter(
        dataset_builder, args.optim_batch_size, shard_shape, split=val_split)
    log_str = (
        f'It will go through {val_steps_per_epoch} steps to handle '
        f'{val_dataset_size} validation data.')
    print_fn(log_str)

    # ----------------------------------------------------------------------- #
    # Load model
    # ----------------------------------------------------------------------- #
    config = None # pylint: disable=invalid-name
    params = None # pylint: disable=invalid-name

    if args.model_name in (
            'openai/clip-vit-large-patch14',
        ):
        from transformerx.models.clip_vision.default import \
            load_jx_config, load_jx_params
        from transformerx.models.clip_vision.modeling import \
            forward_fn as _forward_fn, CLIPVisionInputs

        config = load_jx_config(args.model_name)
        params = load_jx_params(args.model_name)

        # pylint: disable=unnecessary-lambda-assignment
        packing_inputs = lambda input_pixels: \
            CLIPVisionInputs(input_pixels=input_pixels)

    if config is None:
        raise NotImplementedError(f'Unknown args.model_name={args.model_name}')

    # ----------------------------------------------------------------------- #
    # Setup model
    # ----------------------------------------------------------------------- #
    IMAGE_MEAN = jnp.array([[[[0.48145466, 0.45782750, 0.40821073]]]])
    IMAGE_STD = jnp.array([[[[0.26862954, 0.26130258, 0.27577711]]]])

    init_position = {
        'ext': params,
        'cls': jnp.load(f'./examples/finetuning/{args.data_name}.npy')}

    def forward_fn(params, images): # pylint: disable=redefined-outer-name
        """Returns logit vector for each instance."""
        images = (images / 255.0 - IMAGE_MEAN) / IMAGE_STD
        output = _forward_fn(
            params['ext'], packing_inputs(images), config).proj_hidden_states
        output = output / jnp.linalg.norm(output, axis=-1, keepdims=True)
        return output @ params['cls']

    def loss_fn(logits, target):
        """Computes loss for each instance."""
        return jnp.negative(
            jnp.sum(target * jax.nn.log_softmax(logits), axis=-1))

    p_forward_fn = jax.pmap(forward_fn)

    # ----------------------------------------------------------------------- #
    # Fine-tuning model
    # ----------------------------------------------------------------------- #
    @partial(jax.pmap, axis_name='batch')
    def step_trn(state, batch): # pylint: disable=redefined-outer-name
        """Updates state for a given batch."""

        def _scheduler(step): # pylint: disable=redefined-outer-name
            return 0.5 + 0.5 * jnp.cos(
                step.astype(float) / args.optim_num_steps * jnp.pi)

        def _loss_fn(params): # pylint: disable=redefined-outer-name
            logits = forward_fn(params, batch['images'])
            target = jax.nn.one_hot(batch['labels'], logits.shape[-1])
            return jnp.mean(loss_fn(logits, target))

        lr = args.optim_learning_rate * _scheduler(state.step)
        loss, state = optim_step(
            state=state, loss_fn=_loss_fn,
            learning_rate=lr, grad_mask=grad_mask,
            argnums=0, has_aux=False, axis_name='batch')
        metric = OrderedDict({ # pylint: disable=redefined-outer-name
            'step': state.step, 'loss': loss, 'lr': lr})
        return state, metric

    def accuracy(probs, labels):
        """Computes classification accuracy."""
        return float(jnp.mean(jnp.equal(jnp.argmax(probs, axis=-1), labels)))

    def categorical_negative_log_likelihood(probs, labels, eps=1e-06):
        """Computes categorical negative log-likelihood."""
        return float(jnp.mean(jnp.negative(jnp.sum(jax.nn.one_hot(
            labels, probs.shape[-1]) * jnp.log(probs + eps), axis=-1))))

    def _get_metrics(device_metrics):
        return jax.tree_util.tree_map(
            lambda *args: np.stack(args), *jax.device_get(
                jax.tree_util.tree_map(lambda x: x[0], device_metrics)))

    def _mask_fn(param, is_masked):
        if is_masked:
            return jnp.zeros((), param.dtype)
        return param

    grad_mask = []
    path_to_be_updated = []
    path_to_be_freezed = []
    path_and_leaves, treedef \
        = jax.tree_util.tree_flatten_with_path(init_position)
    for path, leave in path_and_leaves:
        if jax.tree_util.DictKey('embeddings') in path:
            path_to_be_freezed.append(path)
            grad_mask.append(True)
        else:
            path_to_be_updated.append(path)
            grad_mask.append(False)

    grad_mask = jax.tree_util.tree_unflatten(treedef, grad_mask)
    log_str = (
        f'Following {len(path_to_be_updated)} parameter groups will be '
        f'updated:\n' + ', '.join(
            jax.tree_util.keystr(e) for e in path_to_be_updated))
    print_fn(log_str)

    log_str = (
        f'Following {len(path_to_be_freezed)} parameter groups will be '
        f'freezed:\n' + ', '.join(
            jax.tree_util.keystr(e) for e in path_to_be_freezed))
    print_fn(log_str)

    if args.optim == 'adam':
        state = adam.AdamState(
            step=0, position=init_position,
            momentum_mu=jax.tree_util.tree_map(
                _mask_fn, jax.tree_util.tree_map(
                    jnp.zeros_like, init_position), grad_mask),
            momentum_nu=jax.tree_util.tree_map(
                _mask_fn, jax.tree_util.tree_map(
                    jnp.zeros_like, init_position), grad_mask))
        optim_step = partial(
            adam.step,
            weight_decay=args.optim_weight_decay,
            clip_radius=args.optim_clip_radius,
            momentums=(args.optim_momentum_mu, args.optim_momentum_nu))

    if args.optim == 'ivon':
        rng_key = jax.random.PRNGKey(args.seed)
        state = ivon.IVONState(
            step=0, rng_key=rng_key, position=init_position,
            momentum_mu=jax.tree_util.tree_map(
                _mask_fn, jax.tree_util.tree_map(
                    jnp.zeros_like, init_position), grad_mask),
            momentum_nu=jax.tree_util.tree_map(
                _mask_fn, jax.tree_util.tree_map(
                    jnp.ones_like, init_position), grad_mask))
        optim_step = partial(
            ivon.step,
            effective_sample_size=trn_dataset_size*args.optim_ess_factor,
            weight_decay=args.optim_weight_decay,
            clip_radius=args.optim_clip_radius,
            momentums=(args.optim_momentum_mu, args.optim_momentum_nu))

    state = jax.device_put_replicated(state, jax.local_devices())

    best_acc = 0.0 # pylint: disable=invalid-name
    trn_metric = []
    for step in range(1, args.optim_num_steps + 1):
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        log_str = f'[Step {step:7d}/{args.optim_num_steps:7d}] '

        batch = next(trn_iter)
        state, metric = step_trn(state, batch)
        trn_metric.append(metric)

        if step == 1 or step % 1000 == 0:
            trn_summarized = jax.tree_util.tree_map(
                lambda e: e.mean(), _get_metrics(trn_metric))
            trn_summarized = {f'trn/{k}': v for k, v in trn_summarized.items()}
            log_str += ', '.join(
                f'{k} {v:.3e}' for k, v in trn_summarized.items())
            trn_metric = []

            logits_list = []
            labels_list = []
            for _ in range(val_steps_per_epoch):
                batch = next(val_iter)
                logits_list.append(jax.device_put(
                    p_forward_fn(state.position, batch['images']).reshape(
                        -1, num_classes), jax.devices('cpu')[0]))
                labels_list.append(jax.device_put(
                    batch['labels'].reshape(-1), jax.devices('cpu')[0]))
            with jax.default_device(jax.devices('cpu')[0]):
                logits_list = jnp.concatenate(logits_list)[:val_dataset_size]
                labels_list = jnp.concatenate(labels_list)[:val_dataset_size]
                val_summarized = {
                    'val/acc': accuracy(
                        jax.nn.softmax(logits_list), labels_list),
                    'val/nll': categorical_negative_log_likelihood(
                        jax.nn.softmax(logits_list), labels_list)}
            log_str += ', ' + ', '.join(
                f'{k} {v:.3e}' for k, v in val_summarized.items())
            print_fn(log_str)

            if val_summarized['val/acc'] > best_acc:
                best_acc = val_summarized['val/acc']
                if args.save:
                    ckpt_path = os.path.join(args.save, 'best_acc')
                    pred_path = os.path.join(args.save, 'predictions')
                    # TODO

            if args.wandb:
                wandb.log({
                    **trn_summarized, **val_summarized,
                    'val/best_acc': best_acc})

            if jnp.isnan(val_summarized['val/nll']):
                break
