"""Topk operator (torch.topk)."""

from dataclasses import dataclass, field
from typing import Any
import math

from .core import Operator, OperatorParams, OpType, Registry


@dataclass
class TopkParams(OperatorParams):
    """Topk parameters."""

    batch_seq: int = field(default=1, metadata={"min": 1, "desc": "批次大小 (B)"})
    dim: int = field(default=768, metadata={"min": 1, "desc": "Feature dimension (D)"})
    k: int = field(default=1, metadata={"min": 1, "desc": "Number of top elements (K)"})

@Registry.register_operator("topk", op_type=OpType.COMPUTE)
class TopkOperator(Operator):
    """Topk operator (finds the K largest elements)."""

    COMPUTE_UNIT = "vector"
    PARAM_CLASS = TopkParams

    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.batch_seq = params.batch_seq if params is not None else kwargs.get("batch_seq", 0)
        self.dim = params.dim if params is not None else kwargs.get("dim", 0)
        self.k = params.k if params is not None else kwargs.get("k", 1)

    @property
    def _num_input_elements(self) -> int:
        return self.batch_seq * self.dim

    @property
    def _num_output_elements(self) -> int:
        return self.batch_seq * self.k

    def get_flops(self) -> float:
        """
        Tsopk: sO(N * log K)。
        """
        if self.k <= 1:
            ops_per_element = 1.0
        else:
            ops_per_element = max(1.0, math.log2(self.k))

        return float(self._num_input_elements * ops_per_element)

    def get_memory_details(self) -> dict[str, int]:
        a_sz = self.quant_config.get_compute_size()
        activation_bytes = (self._num_input_elements + self._num_output_elements) * a_sz
        return self.memory_details(
            activation=activation_bytes
        )