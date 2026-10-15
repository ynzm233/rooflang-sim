"""Sigmoid operator."""

from dataclasses import dataclass, field
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry

_SIGMOID_FLOPS_PER_ELEMENT = 10  # exp + add + div ≈ 10 FLOPs


@dataclass
class SigmoidParams(OperatorParams):
    """Sigmoid parameters."""
    batch_seq: int = field(default=1, metadata={"min": 1, "desc": "批次大小 (B)"})
    dim: int = field(default=768, metadata={"min": 1, "desc": "Feature dimension (D)"})


@Registry.register_operator("sigmoid", op_type=OpType.COMPUTE)
class SigmoidOperator(Operator):
    """Sigmoid activation: y = 1 / (1 + exp(-x))"""

    COMPUTE_UNIT = "vector"
    PARAM_CLASS = SigmoidParams

    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.batch_seq = params.batch_seq if params is not None else kwargs.get("batch_seq", 0)
        self.dim = params.dim if params is not None else kwargs.get("dim", 0)

    @property
    def _num_elements(self) -> int:
        return self.batch_seq * self.dim

    def get_flops(self) -> float:
        return _SIGMOID_FLOPS_PER_ELEMENT * self._num_elements

    def get_memory_details(self) -> dict[str, int]:
        a_sz = self.quant_config.get_compute_size()
        activation_bytes = 2 * self._num_elements * a_sz  # input + output
        return self.memory_details(
            activation=activation_bytes,
        )