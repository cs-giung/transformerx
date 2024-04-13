"""Utilities for handling pytree objects."""
import pickle
from pathlib import Path
from typing import Union

from transformerx.typing import Pytree, PytreeLike


def save(
        file: Union[str, Path],
        pytree: PytreeLike,
        *,
        overwrite: bool = False,
    ) -> None:
    """Save a pytree to a binary file in `.pickle` format.

    Args:
        file: Filename to which the data is saved; a `.pickle` extension will
            be appended to the filename if it does not already ahve one.
        pytree: Pytree data to be saved.
        overwrite: It prohibits overwriting of files (default: False).
    """
    file = Path(file)
    if file.suffix != '.pickle':
        file = file.with_suffix('.pickle')

    if file.exists():
        if overwrite:
            file.unlink()
        else:
            raise RuntimeError(
                f'{file} already exists, while overwrite is {overwrite}.')

    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, 'wb') as fp:
        pickle.dump(pytree, fp)


def load(file: Union[str, Path]) -> Pytree:
    """Load a pytree from a binary file in `.pickle` format.

    Args:
        file: Filename to which the data is saved.
    """
    file = Path(file)
    if not file.is_file():
        raise ValueError(f'{file} is not a file.')
    if file.suffix != '.pickle':
        raise ValueError(f'{file} is not a .pickle file.')
    with open(file, 'rb') as fp:
        pytree = pickle.load(fp)
    return pytree
