"""Fine-tuning"""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import jaxlib
import tensorflow
import tensorflow_datasets
import transformers
from jax_smi import initialise_tracking
initialise_tracking()

from examples.default import get_args, str2bool 
from examples.finetuning.input_pipeline import create_trn_iter, create_val_iter
from transformerx.experimental.optimization import adam


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
        '--optim_learning_rate', default=1e-05, type=float,
        help='a peak learning rate for training (default: 1e-05)')

    args, print_fn = get_args(
        parser, exist_ok=False, dot_log_file=False,
        libraries=(jax, jaxlib, tensorflow, tensorflow_datasets, transformers))

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
            forward_fn, CLIPVisionInputs

        config = load_jx_config(args.model_name)
        params = load_jx_params(args.model_name)
        packing_inputs = lambda input_pixels: \
            CLIPVisionInputs(input_pixels=input_pixels)

    if config is None:
        raise NotImplementedError(f'Unknown args.model_name={args.model_name}')

    # ----------------------------------------------------------------------- #
    # Setup model
    # ----------------------------------------------------------------------- #
    init_position = {
        'ext': params,
        'cls': jnp.zeros((config.projection_dim, num_classes))}

    def forward_fn(params, images):
        return forward_fn(
            params['ext'], packing_inputs(images), config) @ params['cls']

    def loss_fn(logits, target):
        """Computes loss for each instance."""
        return jnp.negative(
            jnp.sum(target * jax.nn.log_softmax(logits), axis=-1))

    # ----------------------------------------------------------------------- #
    # Fine-tuning model
    # ----------------------------------------------------------------------- #
    def get_metrics(device_metrics):
        return jax.tree_util.tree_map(
            lambda *args: np.stack(args), *jax.device_get(
                jax.tree_util.tree_map(lambda x: x[0], device_metrics)))

    @partial(jax.pmap, axis_name='batch')
    def step_trn(state, batch):
        """Updates state for a given batch."""

        def _scheduler(step):
            return 0.5 + 0.5 * jnp.cos(
                step.astype(float) / args.optim_num_steps * jnp.pi)

        def _loss_fn(params):
            logits = forward_fn(params, batch['images'])
            target = jax.nn.one_hot(batch['labels'], logits.shape[-1])
            return jnp.mean(loss_fn(logits, target))

        lr = args.optim_learning_rate * _scheduler(state.step)
        loss, state = optim_step(
            state=state, loss_fn=_loss_fn,
            learning_rate=lr, grad_mask=grad_mask,
            argnums=0, has_aux=False, axis_name='batch')
        metric = OrderedDict({'step': state.step, 'loss': loss, 'lr': lr})
        return state, metric



    state = adam.AdamState(
        step=0, position=init_position,
        momentum_mu=jax.tree_util.tree_map(
            _mask_fn, jax.tree_util.tree_map(jnp.zeros_)
            )
            )



    best_acc = 0.0
    trn_metric = []
    for step in range(1, args.optim_num_steps + 1):
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        log_str = f'[Step {step:7d}/{args.optim_num_steps:7d}] '

        batch = next(trn_iter)
        state, metric = step_trn(state, batch)
        trn_metric.append(metric)

        if step == 1 or step % 1000 == 0:
            trn_summarized = jax.tree_util.tree_map(
                lambda e: e.mean(), get_metrics(trn_metric))
            trn_summarized = {f'trn/{k}': v for k, v in trn_summarized.items()}
            log_str += ', '.join(
                f'{k} {v:.3e}' for k, v in trn_summarized.items())
            trn_metric = []
