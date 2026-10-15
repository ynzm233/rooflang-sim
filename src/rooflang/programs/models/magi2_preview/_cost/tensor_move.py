"""Explicit materialization boundaries for tensor layout and dtype operations."""

from dataclasses import dataclass, field
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry
from .core import DTYPE_INFO, get_dtype_size

TENSOR_MOVE_MODES = ["copy", "permute", "pad", "reshape", "cast", "concat", "crop",
                     "gather", "scatter", "unpatchify", "deinterleave"]


@dataclass
class TensorMoveParams(OperatorParams):
    mode: str = field(default="copy", metadata={"choices": TENSOR_MOVE_MODES})
    input_elements: int = field(default=1, metadata={"min": 0})
    output_elements: int = field(default=1, metadata={"min": 0})
    input_dtype: str | None = None
    output_dtype: str | None = None
    index_elements: int = field(default=0, metadata={"min": 0})
    index_dtype: str = "int64"
    materialized: bool = True

    def validate(self) -> None:
        super().validate()
        if not self.materialized and self.mode not in ("reshape", "permute", "crop"):
            raise ValueError("only reshape, permute and crop can be represented as non-materialized views")
        if self.mode == "cast" and self.input_elements != self.output_elements:
            raise ValueError("cast preserves the number of elements")


@Registry.register_operator("tensor_move", op_type=OpType.COMPUTE)
class TensorMoveOperator(Operator):
    """Count a named copy/layout boundary, not an arbitrary workload formula.

    input_elements sums the input regions actually read; output_elements counts
    materialized output regions. Different lengths represent padding, cropping,
    or indexed selection. Views explicitly set materialized=False and own no
    traffic. Nearest interpolation belongs to InterpolateOperator instead.
    """

    # HBM tensor traffic uses the normal compute roofline. The library's
    # "storage" unit is reserved for persistent KV-cache hierarchy loads.
    COMPUTE_UNIT = "vector"
    PARAM_CLASS = TensorMoveParams

    def parse_args(self, **kwargs: Any) -> None:
        self.params.validate()
        for name in self.params.__dataclass_fields__:
            setattr(self, name, getattr(self.params, name))
        self.input_dtype = self.input_dtype or self.compute_dtype
        self.output_dtype = self.output_dtype or self.compute_dtype
        if self.input_dtype not in DTYPE_INFO or self.output_dtype not in DTYPE_INFO:
            raise ValueError("unsupported tensor boundary dtype")
        if self.index_dtype not in ("int8", "int32", "int64"):
            raise ValueError("index_dtype must be int8, int32 or int64")
        self._input_size = get_dtype_size(self.input_dtype)
        self._output_size = get_dtype_size(self.output_dtype)
        # The shared precision catalog contains int32 but may not contain int64.
        self._index_size = 8 if self.index_dtype == "int64" else get_dtype_size(self.index_dtype)
        self.shape = (f"{self.mode}: [{self.input_elements}]/{self.input_dtype} -> "
                      f"[{self.output_elements}]/{self.output_dtype}; materialized={self.materialized}")
        self.assumption = "logical materialization traffic; layout index arithmetic and conversion throughput excluded"

    def get_flops(self) -> float:
        return 0.0

    def get_memory_details(self) -> dict[str, int]:
        traffic = (self.input_elements * self._input_size + self.output_elements * self._output_size
                   + self.index_elements * self._index_size) if self.materialized else 0
        return self.memory_details(activation=traffic)
