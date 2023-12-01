"""Generic utilities."""
from __future__ import annotations
import itertools
from typing import List, TypeVar
T = TypeVar('T')


def chunks(iterable: List[T], size) -> List[List[T]]:
    if size == 0:
        raise ValueError('Chunk size should not be zero.')
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk
