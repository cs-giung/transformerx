"""SymmetricQuantizedArray built on Qax."""
from dataclasses import dataclass
from itertools import chain
from typing import Sequence

import jax
import jax.numpy as jnp
from qax import aux_field, ImplicitArray, primitive_handler
from qax.implicit.implicit_array import \
    _aval_discovery, _get_names_and_aux, UninitializedAval
from transformerx.typing import Array, ArrayLike


@dataclass
class SymmetricQuantizedArray(ImplicitArray):
    """Represents a symmetrically quantized array."""
    # pylint: disable=too-many-instance-attributes
    q_value: ArrayLike
    scale_factor: ArrayLike
    contraction_axis: int = aux_field()
    group_size: int = aux_field()
    bits: int = aux_field()
    shape: Sequence[int] = aux_field()
    dtype: jnp.dtype = aux_field()
    itype: jnp.dtype = aux_field()

    @staticmethod
    @jax.default_device(jax.devices('cpu')[0])
    def quantize(
            arr: ArrayLike,
            *,
            bits: int,
            contraction_axis: int,
            group_size: int,
        ):
        """Returns quantized array."""

        # TODO: it now uses int8 even for smaller bits.
        itype = jnp.int8
        dtype = arr.dtype
        shape = arr.shape

        arr = arr.transpose(*chain(
            range(contraction_axis),
            range(contraction_axis + 1, len(arr.shape)),
            [contraction_axis]))
        arr = arr.reshape(-1, group_size, arr.shape[-1])

        max_q = 2 ** (bits - 1) - 1
        max_x = jnp.max(jnp.abs(arr), axis=(1, 2))
        scale_factor = (max_x / max_q).astype(dtype)

        q_value = arr / scale_factor[:, None, None]
        q_value = jnp.round(q_value)
        q_value = jnp.clip(q_value, -max_q, max_q).astype(itype)

        # pylint: disable=too-many-function-args
        return SymmetricQuantizedArray(
            q_value=q_value, scale_factor=scale_factor,
            contraction_axis=contraction_axis, group_size=group_size,
            bits=bits, shape=shape, dtype=dtype, itype=itype)

    @staticmethod
    @jax.default_device(jax.devices('cpu')[0])
    def dequantize(
            q_value: ArrayLike,
            scale_factor: ArrayLike,
            contraction_axis: int,
            shape: Sequence[int],
        ) -> Array:
        """Returns dequantized array."""

        arr = q_value * scale_factor[:, None, None]
        arr = arr.reshape(
            shape[:contraction_axis]
            + shape[contraction_axis+1:]
            + (shape[contraction_axis],))
        arr = arr.transpose(*chain(
            range(contraction_axis),
            [arr.ndim - 1],
            range(contraction_axis, arr.ndim - 1)))

        return arr

    def materialize(self) -> Array:
        return self.dequantize(
            self.q_value, self.scale_factor,
            self.contraction_axis, self.shape)

    # TODO: any problem?
    def tree_flatten_with_keys(self):
        children = []
        aux_data = []
        for name, is_aux in _get_names_and_aux(self):
            try:
                value = getattr(self, name)
            except UninitializedAval:
                if not _aval_discovery.get():
                    raise
                value = None
            if is_aux:
                aux_data.append(value)
            else:
                children.append((jax.tree_util.GetAttrKey(name), value))
        return children, aux_data


def _check_dot_general_dimension_numbers(dimension_numbers):
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    if lhs_batch or rhs_batch:
        return False
    if len(lhs_contract) != 1 or len(rhs_contract) != 1:
        return False
    return True


def dot_general(
        lhs: SymmetricQuantizedArray, rhs: SymmetricQuantizedArray,
        dimension_numbers, precision=None, preferred_element_type=jnp.int32,
    ) -> Array:
    """"General dot product operator."""

    # TODO:
    if not _check_dot_general_dimension_numbers(dimension_numbers):
        raise NotImplemented
    if lhs.group_size != 1:
        raise NotImplemented

    lhs_q_value = lhs.q_value.reshape(
        lhs.shape[:lhs.contraction_axis]
        + lhs.shape[lhs.contraction_axis+1:]
        + (lhs.shape[lhs.contraction_axis],))
    lhs_q_value = lhs_q_value.transpose(*chain(
        range(lhs.contraction_axis),
        [lhs_q_value.ndim - 1],
        range(lhs.contraction_axis, lhs_q_value.ndim - 1)))

    rhs_q_value = rhs.q_value.reshape(
        rhs.shape[:rhs.contraction_axis]
        + rhs.shape[rhs.contraction_axis+1:]
        + (rhs.shape[rhs.contraction_axis],))
    rhs_q_value = rhs_q_value.transpose(*chain(
        range(rhs.contraction_axis),
        [rhs_q_value.ndim - 1],
        range(rhs.contraction_axis, rhs_q_value.ndim - 1)))

    res = jax.lax.dot_general(
        lhs_q_value, rhs_q_value, dimension_numbers=dimension_numbers,
        precision=precision, preferred_element_type=preferred_element_type)

    return res * lhs.scale_factor[:, None] \
        * jnp.repeat(rhs.scale_factor, rhs.group_size)[None, :]
