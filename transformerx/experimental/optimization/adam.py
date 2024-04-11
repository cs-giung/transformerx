"""Adam optimizer."""
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from transformerx.typing import Array, PytreeLike


class AdamState(NamedTuple):
    step: int
    position: PytreeLike
    momentum_mu: PytreeLike
    momentum_nu: PytreeLike


def step(
        state: AdamState,
        loss_fn: Callable[..., float],
        learning_rate: float,
        momentums: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
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
        momentums
        eps
        grad_mask
        argnums
        has_aux
        axis_name

    Returns:
        a tuple of output
    """
    grad_fn = jax.value_and_grad(loss_fn, argnums, has_aux)

    aux, gradient = grad_fn(state.position)
    if axis_name is not None:
        gradient = jax.lax.pmean(gradient, axis_name)

    if grad_mask is not None:
        def _mask_fn(param, is_masked):
            if is_masked:
                return jnp.zeros((), param.dtype)
            return param
        gradient = jax.tree_util.tree_map(_mask_fn, gradient, grad_mask)

    new_mu = jax.tree_util.tree_map(
        lambda m, g: m * momentums[0] + g**1 * (1.0 - momentums[0]),
        state.momentum_mu, gradient)
    new_nu = jax.tree_util.tree_map(
        lambda n, g: n * momentums[1] + g**2 * (1.0 - momentums[1]),
        state.momentum_nu, gradient)

    mu_hat = jax.tree_util.tree_map(
        lambda m: m / (1.0 - momentums[0]**(state.step + 1)), new_mu)
    nu_hat = jax.tree_util.tree_map(
        lambda n: n / (1.0 - momentums[1]**(state.step + 1)), new_nu)
    new_position = jax.tree_util.tree_map(
        lambda p, m, n: p - learning_rate * m / (jnp.sqrt(n) + eps),
        state.position, mu_hat, nu_hat)

    return aux, AdamState(
        step=state.step+1, position=new_position,
        momentum_mu=new_mu, momentum_nu=new_nu)
