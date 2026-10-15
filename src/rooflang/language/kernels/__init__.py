# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Kernel classes for closed-form inference FLOPs and HBM-byte formulas.

  kernel.py     — Kernel base class + TensorDesc
  forward.py    — Slice, Gemm, StridedGemm, RMSNorm, LayerNorm,
                  Attn, routing, and dependency kernels
  comm.py       — AllReduce, ReduceScatter, AllGather, AllToAll, Broadcast, SendRecv
  identity.py   — Spawn, Concat

Example:
    from rooflang.language.kernels import forward
    fwd = forward.Gemm(4096, 4096, 4096, 'fp8', 'fp8')
"""

from rooflang.language.kernels.kernel import Kernel  # noqa: F401
from rooflang.language.tensor import Tensor  # noqa: F401
