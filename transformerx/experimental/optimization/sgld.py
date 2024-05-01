"""SGLD sampler."""
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from transformerx.typing import Array, PRNGKeyLike, PytreeLike


Scalar = Any


class SGLDState(NamedTuple): # pylint: disable=missing-class-docstring
    step: int
    rng_key: PRNGKeyLike
    position: PytreeLike


def step(
        state: SGLDState,
        loss_fn: Callable[..., Scalar],
        learning_rate: Scalar,
        weight_decay: Scalar,
        num_data: Scalar,
        posterior_temperature: Scalar,
        clip_radius: Scalar = None,
        grad_mask: PytreeLike = None,
        argnums: Union[int, Tuple[int, ...]] = 0,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
    ) -> Tuple[Union[Tuple[Array, Any], Array], SGLDState]:
    """Step function for SGLD."""
    # pylint: disable=too-many-arguments
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

    tree = jax.tree_util.tree_structure(grad)
    keys = jax.tree_util.tree_unflatten(
        tree, jax.random.split(state.rng_key, tree.num_leaves))
    _updates = jax.tree_util.tree_map(
        lambda g, k: g + jnp.sqrt(
            2.0 * (learning_rate / num_data) * posterior_temperature
        ) * jax.random.normal(k, g.shape, g.dtype), grad, keys)

    if clip_radius:
        _updates = jax.tree_util.tree_map(
            lambda e: jnp.clip(e, -clip_radius, clip_radius), _updates)
    if grad_mask:
        _updates = mask_fn(_updates)

    new_position = jax.tree_util.tree_map(
        lambda p, u: p - learning_rate * u,
        state.position, _updates)

    new_position = jax.tree_util.tree_map(
        lambda np, p: np - learning_rate * weight_decay * p,
        new_position, state.position)

    new_rng_key = jax.random.split(state.rng_key)[0]

    return aux, SGLDState(
        step=state.step+1, rng_key=new_rng_key, position=new_position)
