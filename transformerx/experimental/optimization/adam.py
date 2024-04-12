"""Adam optimizer."""
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from transformerx.typing import Array, PytreeLike


Scalar = Any


class AdamState(NamedTuple): # pylint: disable=missing-class-docstring
    step: int
    position: PytreeLike
    momentum_mu: PytreeLike
    momentum_nu: PytreeLike


def step(
        state: AdamState,
        loss_fn: Callable[..., Scalar],
        learning_rate: Scalar,
        weight_decay: Scalar,
        clip_radius: Scalar = None,
        momentums: Tuple[Scalar, Scalar] = (0.9, 0.999),
        eps: Scalar = 1e-08,
        grad_mask: PytreeLike = None,
        argnums: Union[int, Tuple[int, ...]] = 0,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
    ) -> Tuple[Union[Tuple[Array, Any], Array], AdamState]:
    """Step function for Adam.

    We follow the algorithm described in Kingma and Ba (2015).

    Args:
        state
        loss_fn
        learning_rate
        weight_decay
        clip_radius
        momentums
        eps
        grad_mask
        argnums
        has_aux
        axis_name

    Returns:
        a tuple of output
    """
    # pylint: disable=too-many-arguments,too-many-locals
    def mask_fn(pytree):
        def _mask_fn(param, is_masked):
            if is_masked:
                return jnp.zeros((), param.dtype)
            return param
        return jax.tree_util.tree_map(_mask_fn, pytree, grad_mask)

    grad_fn = jax.value_and_grad(loss_fn, argnums, has_aux)
    aux, grad = grad_fn(state.position)
    if axis_name is not None:
        grad = jax.lax.pmean(grad, axis_name)
    if grad_mask is not None:
        grad = mask_fn(grad)

    new_mu = jax.tree_util.tree_map(
        lambda mu, g: mu * momentums[0] + g**1 * (1.0 - momentums[0]),
        state.momentum_mu, grad)
    new_nu = jax.tree_util.tree_map(
        lambda nu, g: nu * momentums[1] + g**2 * (1.0 - momentums[1]),
        state.momentum_nu, grad)

    mu_hat = jax.tree_util.tree_map(
        lambda mu: mu / (1.0 - momentums[0]**(state.step + 1)), new_mu)
    nu_hat = jax.tree_util.tree_map(
        lambda nu: nu / (1.0 - momentums[1]**(state.step + 1)), new_nu)

    _updates = jax.tree_util.tree_map(
        lambda mu, nu: mu / (jnp.sqrt(nu) + eps), mu_hat, nu_hat)
    if clip_radius:
        _updates = jax.tree_util.tree_map(
            lambda e: jnp.clip(e, -clip_radius, clip_radius), _updates)
    if grad_mask:
        _updates = mask_fn(_updates)

    new_position = jax.tree_util.tree_map(
        lambda p, u: p - learning_rate * u, state.position, _updates)

    new_position = jax.tree_util.tree_map(
        lambda np, p: np - learning_rate * weight_decay * p,
        new_position, state.position)

    return aux, AdamState(
        step=state.step+1, position=new_position,
        momentum_mu=new_mu, momentum_nu=new_nu)
