"""IVON optimizer."""
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from transformerx.typing import Array, PRNGKeyLike, PytreeLike


Scalar = Any


class IVONState(NamedTuple): # pylint: disable=missing-class-docstring
    step: int
    rng_key: PRNGKeyLike
    position: PytreeLike
    momentum_mu: PytreeLike
    momentum_nu: PytreeLike


def step(
        state: IVONState,
        loss_fn: Callable[..., Scalar],
        learning_rate: Scalar,
        effective_sample_size: Scalar,
        weight_decay: Scalar,
        clip_radius: Scalar = None,
        momentums: Tuple[Scalar, Scalar] = (0.9, 0.99999),
        grad_mask: PytreeLike = None,
        argnums: Union[int, Tuple[int, ...]] = 0,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
    ) -> Tuple[Union[Tuple[Array, Any], Array], IVONState]:
    """Step function for IVON.

    We follow the algorithm described in Shen et al. (2024).

    Args:
        state: the current optimization state.
        loss_fn: the loss function to be differentiated; it should take
            arguments at positions specified by `argnums`, which can be arrys,
            scalars, or common Python containers; the function should return a
            scalar, including arrays with sahpe `()`, but not arrays with other
            shapes like `(1,)`.
        learning_rate: a float learning rate value.
        effective_sample_size: setting it to the size of training dataset
            recovers the standard evidence lower bound objective for
            variational learning; setting it smaller than the size of training
            dataset is equivalent to increased temperature and setting it
            higher to decreased temperature.
        weight_decay: a float value for weight decay regularization.
        clip_radius: a float value for clipping the update.
        momentums: a tuple of float momentum factors for computing running
            averages of gradient and hessian (default: (0.9, 0.99999)).
        grad_mask: a pytree to mask gradient; it should have the same tree
            structure to that of `state.position` (default: None).
        argnums: an integer or a sequence of intergers; it dermines which
            positional argument(s) to differentiate with (default: 0).
        has_aux: it indicates whether the `loss_fn` returns a pair, with the
            first element as the main output of the loss function for
            differentiation and the second element as optional auxiliary data
            (default: False).
        axis_name: when an `axis_name` is provided, the gradient will be
            averaged across replicas (default: None).

    Returns:
        a tuple of `loss_fn` outputs and the updated optimization state.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    def mask_fn(pytree):
        def _mask_fn(param, is_masked):
            if is_masked:
                return jnp.zeros((), param.dtype)
            return param
        return jax.tree_util.tree_map(_mask_fn, pytree, grad_mask)

    def randn_like(rng_key, pytree):
        treedef = jax.tree_util.tree_structure(pytree)
        rng_key = jax.tree_util.tree_unflatten(
            treedef, jax.random.split(rng_key, treedef.num_leaves))
        return jax.tree_util.tree_map(
            lambda p, k: jax.random.normal(k, p.shape, p.dtype),
            pytree, rng_key)

    noise = randn_like(state.rng_key, state.position)
    if grad_mask is not None:
        noise = mask_fn(noise)
    noisy_position = jax.tree_util.tree_map(
        lambda p, n, nu: p + n * jnp.sqrt(
            1.0 / (effective_sample_size * (nu + weight_decay))),
        state.position, noise, state.momentum_nu)

    grad_fn = jax.value_and_grad(loss_fn, argnums, has_aux)
    aux, grad = grad_fn(noisy_position)
    if axis_name is not None:
        grad = jax.lax.pmean(grad, axis_name)
    if grad_mask is not None:
        grad = mask_fn(grad)

    hess = jax.tree_util.tree_map(
        lambda g, np, p, nu: g * (np - p) \
            * effective_sample_size * (nu + weight_decay),
        grad, noisy_position, state.position, state.momentum_nu)
    new_mu = jax.tree_util.tree_map(
        lambda mu, g: mu * momentums[0] + g * (1.0 - momentums[0]),
        state.momentum_mu, grad)
    new_nu = jax.tree_util.tree_map(
        lambda nu, h: nu * momentums[1] + h * (1.0 - momentums[1]),
        state.momentum_nu, hess)

    mu_hat = jax.tree_util.tree_map(
        lambda mu: mu / (1.0 - momentums[0] ** (state.step + 1)), new_mu)
    nu_hat = new_nu

    _updates = jax.tree_util.tree_map(
        lambda p, mu, nu: (mu + weight_decay * p) / (nu + weight_decay),
        state.position, mu_hat, nu_hat)
    if clip_radius:
        _updates = jax.tree_util.tree_map(
            lambda e: jnp.clip(e, -clip_radius, clip_radius), _updates)
    if grad_mask:
        _updates = mask_fn(_updates)

    new_position = jax.tree_util.tree_map(
        lambda p, u: p - learning_rate * u, state.position, _updates)
    new_rng_key = jax.random.split(state.rng_key)[0]

    return aux, IVONState(
        step=state.step+1, rng_key=new_rng_key, position=new_position,
        momentum_mu=new_mu, momentum_nu=new_nu)
