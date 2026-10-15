"""Softmax operator."""

from dataclasses import dataclass, field
from typing import Any, List

from .core import Operator, OperatorParams, OpType, Registry

# Softmax FLOPs: exp(1) + sub(1) + div(1) + reductions(2) = 5
_SOFTMAX_FLOPS_PER_ELEMENT = 5


@dataclass
class SoftmaxParams(OperatorParams):
    """Softmax parameters."""
    batch_seq: int = field(default=1, metadata={"min": 1, "desc": "批次大小 (B)"})
    dim: int = field(default=768, metadata={"min": 1, "desc": "Feature dimension (D)"})
    fused: List[bool] = field(default_factory=lambda: [False, False], metadata={"desc": "是否被融合"})

@Registry.register_operator("softmax", op_type=OpType.COMPUTE)
class SoftmaxOperator(Operator):
    """Softmax operator."""

    COMPUTE_UNIT = "vector"
    PARAM_CLASS = SoftmaxParams

    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.batch_seq = params.batch_seq if params is not None else kwargs.get("batch_seq", 0)
        self.dim = params.dim if params is not None else kwargs.get("dim", 0)
        self.fused = params.fused if params is not None else kwargs.get("fused", [False, False])

    @property
    def _num_elements(self) -> int:
        return self.batch_seq * self.dim

    def get_flops(self) -> float:
        return float(self._num_elements * _SOFTMAX_FLOPS_PER_ELEMENT)

    def get_memory_details(self) -> dict[str, int]:
        input_size = 0 if self.fused[0] else self._num_elements
        output_size = 0 if self.fused[1] else self._num_elements
        activation_bytes = (input_size + output_size) * self.quant_config.get_compute_size()
        return self.memory_details(activation=activation_bytes)
