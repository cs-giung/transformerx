"""LoRA utilities built on Qax."""
from dataclasses import dataclass
from typing  import Callable

import jax
import jax.numpy as jnp
from qax import aux_field, ImplicitArray
from qax.implicit.implicit_array import \
    _aval_discovery, _get_names_and_aux, UninitializedAval
from transformerx.typing import Array, ArrayLike, PRNGKeyLike


@dataclass
class LoraArray(ImplicitArray):
    """Represents a LoRA array."""
    lora_w: ArrayLike
    lora_a: ArrayLike
    lora_b: ArrayLike
    lora_rank: int = aux_field()
    lora_alpha: int = aux_field(default=1)

    @staticmethod
    @jax.default_device(jax.devices('cpu')[0])
    def loraize(
            rng_key: PRNGKeyLike,
            arr: ArrayLike,
            *,
            rank: int,
            alpha: int = 1,
            a_init: Callable = jax.nn.initializers.normal(stddev=0.01),
            b_init: Callable = jax.nn.initializers.zeros,
        ):
        """Returns loraized array."""

        # TODO: we now considser matrices only
        assert arr.ndim == 2

        keys = jax.random.split(rng_key)
        lora_a = a_init(keys[0], shape=(rank, arr.shape[1]), dtype=arr.dtype)
        lora_b = b_init(keys[1], shape=(arr.shape[0], rank), dtype=arr.dtype)

        return LoraArray(
            lora_w=arr, lora_a=lora_a, lora_b=lora_b,
            lora_rank=rank, lora_alpha=alpha)

    @property
    def lora_scale(self) -> float:
        return self.lora_alpha / self.lora_rank

    def materialize(self) -> Array:
        return self.lora_w \
            + self.lora_scale * jnp.matmul(self.lora_b, self.lora_a)

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
