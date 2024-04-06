"""It provides JAX-specific static type annotations."""
from typing import Any, Iterable, Mapping, Union # pylint: disable=import-self

# JAX Typing
# https://jax.readthedocs.io/en/latest/jax.typing.html
from jax import Array
from jax.typing import ArrayLike

# JAX PRNG Keys
# https://jax.readthedocs.io/en/latest/jax.random.html
PRNGKey = Array
PRNGKeyLike = ArrayLike

# JAX Pytrees
# https://jax.readthedocs.io/en/latest/pytrees.html
Pytree = Union[
    Array, Iterable["Pytree"], Mapping[Any, "Pytree"]]
PytreeLike = Union[
    ArrayLike, Iterable["PytreeLike"], Mapping[Any, "PytreeLike"]]
