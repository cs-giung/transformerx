"""This module implements einshard."""
from math import prod

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformerx.experimental.einshard.parser import parse_expression
from transformerx.typing import Array, ArrayLike


def einshard(arr: ArrayLike, expression: str) -> Array:
    """Shards an array according to a specified expression."""
    # pylint: disable=too-many-locals
    n_devices = jax.device_count()

    res = parse_expression(expression, 0)
    if not res.is_success():
        idx, desc = res.error
        raise ValueError(
            f'Cannot parse einshard expression "{expression}", '
            f'expected {desc} at position {idx}.')
    _, (elements_left, elements_right) = res.value

    n_left_ellipses = sum(
        element_left is ... for element_left in elements_left)
    n_right_ellipses = sum(
        element_right is ... for element_right in elements_right)
    assert n_left_ellipses == n_right_ellipses and n_left_ellipses <= 1

    if n_left_ellipses > 0:
        n_dims = len(arr.shape)
        n_dims_elided = n_dims - len(elements_left) + 1
        axis_names_for_left_augmented = \
            [f'?{i}' for i in range(n_dims_elided)]
        axis_names_for_right_augmented = \
            [(item, 0) for item in axis_names_for_left_augmented]

        def _partition_at_ellipsis(lst: list) -> tuple[list, list]:
            idx = lst.index(...)
            return lst[:idx], lst[idx+1:]

        _l, _r = _partition_at_ellipsis(elements_left)
        elements_left = [*_l, *axis_names_for_left_augmented, *_r]

        _l, _r = _partition_at_ellipsis(elements_right)
        elements_right = [*_l, *axis_names_for_right_augmented, *_r]

    sharding_numbers = [
        integer for _, integer in elements_right if integer != 0]
    n_devices_base = prod(sharding_numbers)
    n_sharded_axes = len(sharding_numbers)
    assert n_devices % n_devices_base == 0

    sharding_ratio_full = n_devices // n_devices_base
    sharding_ratio_one = sharding_ratio_full ** (1. / n_sharded_axes)
    assert sharding_ratio_one.is_integer()
    sharding_ratio_one = int(sharding_ratio_one)

    mesh_shape = [
        1 if integer == 0 else integer * sharding_ratio_one
        for _, integer in elements_right]
    axis_names = tuple(f'a{i}' for i, _ in enumerate(elements_right))
    d = {identifier: i for i, (identifier, _) \
            in enumerate(elements_right) if identifier is not None}
    partition_spec = tuple(
        f'a{d[element_left]}' for element_left in elements_left)

    devices = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices, axis_names=axis_names)
    arr = jax.make_array_from_callback(
        arr.shape, NamedSharding(mesh, PartitionSpec(*partition_spec)),
        lambda idx: arr[idx])
    return arr
