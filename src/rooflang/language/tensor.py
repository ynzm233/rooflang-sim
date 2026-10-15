# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Tensor — a shaped, typed array descriptor."""

from __future__ import annotations

from fractions import Fraction
from math import isfinite
from typing import Tuple

from rooflang.language.utils import dtype_bytes


class Tensor:
    """Descriptor for a tensor slot (input, weight, or output)."""

    def __init__(self, dtype: str, shape: Tuple[int | Fraction, ...],
                 weight_id: str = None, *, storage_bytes: float | None = None) -> None:
        if storage_bytes is not None and (not isfinite(storage_bytes) or storage_bytes < 0):
            raise ValueError("storage_bytes must be finite and non-negative")
        self.dtype = dtype
        self.shape = shape
        self.weight_id = weight_id
        self._storage_bytes = storage_bytes

    @property
    def n_elements(self) -> int | Fraction:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def size_bytes(self) -> float:
        return self.n_elements * dtype_bytes(self.dtype)

    @property
    def storage_bytes(self) -> float:
        """Allocated storage; traffic descriptors may represent repeated reads.

        Ordinary tensors allocate their full size. Analytical aggregate traffic
        can opt out of allocation while retaining size_bytes for bandwidth cost.
        Its actual working set must then be accounted for by the model.
        """
        return self.size_bytes if self._storage_bytes is None else self._storage_bytes
