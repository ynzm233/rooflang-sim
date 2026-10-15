"""Clipped SwiGLU activation used outside the fused MH-MoE boundary."""

from dataclasses import dataclass, field
import math
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry
from .core import DTYPE_INFO, get_dtype_size

SWIGLU7_BASE_FLOPS_PER_ELEMENT = 7.0  # clamp3, multiply3, add1
DEFAULT_SIGMOID_FLOPS = 10.0  # Existing SigmoidOperator convention (Silu uses a separate proxy).


def swiglu7_flops(elements: int, sigmoid_flops: float = DEFAULT_SIGMOID_FLOPS) -> float:
    if elements < 0 or not math.isfinite(sigmoid_flops) or sigmoid_flops < 0:
        raise ValueError("SwiGLU7 counts and sigmoid cost must be finite and nonnegative")
    return float(elements * (SWIGLU7_BASE_FLOPS_PER_ELEMENT + sigmoid_flops))


@dataclass
class SwiGLU7Params(OperatorParams):
    batch_seq: int = field(default=1, metadata={"min": 1})
    dim: int = field(default=1, metadata={"min": 1, "desc": "Output width; input width is 2*dim"})
    sigmoid_flops: float = field(default=DEFAULT_SIGMOID_FLOPS, metadata={"min": 0})
    input_dtype: str | None = None
    output_dtype: str | None = None


@Registry.register_operator("swiglu7", op_type=OpType.COMPUTE)
class SwiGLU7Operator(Operator):
    """gate*sigmoid(1.702*gate)*(up+1), with gate<=7 and -7<=up<=7.

    Each clamp comparison counts as one scalar operation. Input/output traffic
    is one fused activation boundary; no intermediate clamp tensors are billed.
    """

    COMPUTE_UNIT = "vector"
    PARAM_CLASS = SwiGLU7Params

    def parse_args(self, **kwargs: Any) -> None:
        self.params.validate()
        self.batch_seq, self.dim = self.params.batch_seq, self.params.dim
        self.sigmoid_flops = self.params.sigmoid_flops
        self.input_dtype = self.params.input_dtype or self.compute_dtype
        self.output_dtype = self.params.output_dtype or self.compute_dtype
        if self.input_dtype not in DTYPE_INFO or self.output_dtype not in DTYPE_INFO:
            raise ValueError("unsupported SwiGLU7 boundary dtype")
        self._input_size = get_dtype_size(self.input_dtype)
        self._output_size = get_dtype_size(self.output_dtype)
        swiglu7_flops(0, self.sigmoid_flops)
        self.shape = f"[{self.batch_seq},{2*self.dim}] -> [{self.batch_seq},{self.dim}]"
        self.assumption = "fused activation boundary; clamp comparison=1; sigmoid cost is an explicit proxy"

    def get_flops(self) -> float:
        return swiglu7_flops(self.batch_seq * self.dim, self.sigmoid_flops)

    def get_memory_details(self) -> dict[str, int]:
        return self.memory_details(activation=self.batch_seq * self.dim
                                   * (2 * self._input_size + self._output_size))
