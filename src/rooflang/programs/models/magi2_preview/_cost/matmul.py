"""Matmul Projection operator."""

from dataclasses import dataclass, field
from typing import Any, Optional

from .core import Operator, OperatorParams, OpType, Registry

# Matmul FLOPs calculation constant:
# Y = X * W^T + b
# Per output element: D_in multiplications + (D_in - 1) additions + (1 if bias)
# Approximated as: 2 * D_in + (1 if bias)


@dataclass
class MatmulParams(OperatorParams):
    """Matmul projection parameters."""
    """
    - Input x: (B * S * D_in)
    - Output y: (B * S * D_out)
    - weight: (D_in, D_out)
    """
    batch_seq: int = field(default=1, metadata={"min": 1, "desc": "批次大小 (B)"})
    in_features: int = field(default=0, metadata={"min": 1, "desc": "Input dimension (D_in)"})
    out_features: int = field(default=0, metadata={"min": 1, "desc": "Output dimension (D_out)"})
    has_bias: bool = field(default=False, metadata={"desc": "Whether to add bias"})


@Registry.register_operator("matmul", op_type=OpType.COMPUTE)
class MatmulOperator(Operator):
    """Matmul Projection operator (Matrix Multiplication)."""

    COMPUTE_UNIT = "cube"
    PARAM_CLASS = MatmulParams

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.batch_seq = params.batch_seq if params is not None else kwargs.get("batch_seq", 0)
        self.in_features = params.in_features if params is not None else kwargs.get("in_features", 0)
        self.out_features = params.out_features if params is not None else kwargs.get("out_features", 0)
        self.has_bias = params.has_bias if params is not None else kwargs.get("has_bias", False)

    @property
    def _num_output_elements(self) -> int:
        return self.batch_seq * self.out_features

    @property
    def _num_input_elements(self) -> int:
        return self.batch_seq * self.in_features

    @property
    def _num_weight_elements(self) -> int:
        return self.out_features * self.in_features

    def get_flops(self) -> float:
        matmul_ops = 2 * self.in_features
        bias_ops = 1 if self.has_bias else 0

        total_ops_per_element = matmul_ops + bias_ops
        return float(self._num_output_elements * total_ops_per_element)

    def get_memory_details(self) -> dict[str, int]:
        a_sz = self.quant_config.get_compute_size()
        w_sz = self.quant_config.get_weight_size()

        weight_bytes = self._num_weight_elements * w_sz
        if self.has_bias:
            weight_bytes += self.out_features * w_sz

        input_bytes = self._num_input_elements * a_sz
        output_bytes = self._num_output_elements * a_sz
        activation_bytes = input_bytes + output_bytes

        return self.memory_details(
            weight=weight_bytes,
            activation=activation_bytes
        )