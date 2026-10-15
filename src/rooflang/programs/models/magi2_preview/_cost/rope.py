"""Rotary Positional Embedding (RoPE) operator."""

from dataclasses import dataclass, field
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry

# RoPE FLOPs per element estimation:
# For every pair of elements (x, y):
# x' = x * cos - y * sin  (2 mul, 1 sub)
# y' = x * sin + y * cos  (2 mul, 1 add)
# Total: 4 muls + 2 adds = 6 FLOPs for 2 elements.
# Average: 3 FLOPs per element.
# Note: This assumes sin/cos values are pre-computed and loaded from memory/cache.
# If computed on-the-fly via exp/log, FLOPs would be significantly higher.
_ROPE_FLOPS_PER_ELEMENT = 3.0


@dataclass
class RopeParams(OperatorParams):
    """RoPE parameters."""

    batch_seq: int = field(default=1, metadata={"min": 1, "desc": "批次大小 (B)"})
    dim: int = field(default=768, metadata={"min": 1, "desc": "Feature dimension (D)"})

@Registry.register_operator("rope", op_type=OpType.COMPUTE)
class RopeOperator(Operator):
    """Rotary Positional Embedding operator."""

    COMPUTE_UNIT = "vector"
    PARAM_CLASS = RopeParams

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def parse_args(self, **kwargs: Any) -> None:
        """Parse arguments from params or kwargs."""
        params = self.params
        self.batch_seq = params.batch_seq if params is not None else kwargs.get("batch_seq", 0)
        self.dim = params.dim if params is not None else kwargs.get("dim", 0)

    @property
    def _num_elements(self) -> int:
        return self.batch_seq * self.dim

    def get_flops(self) -> float:
        return float(self._num_elements * _ROPE_FLOPS_PER_ELEMENT)

    def get_memory_details(self) -> dict[str, int]:

        activation_bytes = 2 * self._num_elements * self.quant_config.get_compute_size()

        return self.memory_details(
            activation=activation_bytes
        )