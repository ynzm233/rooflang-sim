"""Shape-driven nearest-exact and trilinear interpolation."""

from dataclasses import dataclass, field
import math
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry
from .core import DTYPE_INFO, get_dtype_size


@dataclass
class InterpolateParams(OperatorParams):
    input_shape: list[int] | tuple[int, ...] = field(default_factory=lambda: [1, 1, 1])
    output_shape: list[int] | tuple[int, ...] = field(default_factory=lambda: [1, 1, 1])
    mode: str = field(default="nearest-exact", metadata={"choices": ["nearest-exact", "trilinear"]})
    align_corners: bool | None = None
    input_dtype: str | None = None
    output_dtype: str | None = None

    def validate(self) -> None:
        super().validate()
        for shape in (self.input_shape, self.output_shape):
            if len(shape) not in (3, 4, 5) or any(type(dim) is not int or dim <= 0 for dim in shape):
                raise ValueError("interpolate requires positive [N,C,spatial...] shapes of rank 3..5")
        if len(self.input_shape) != len(self.output_shape) or tuple(self.input_shape[:2]) != tuple(self.output_shape[:2]):
            raise ValueError("interpolate must preserve rank, batch and channels")
        if self.mode == "trilinear" and len(self.input_shape) != 5:
            raise ValueError("trilinear interpolation requires [N,C,D,H,W]")
        if self.mode == "nearest-exact" and self.align_corners is not None:
            raise ValueError("align_corners is only applicable to trilinear interpolation")
        if self.align_corners is not None and type(self.align_corners) is not bool:
            raise ValueError("align_corners must be a boolean or None")


@Registry.register_operator("interpolate", op_type=OpType.COMPUTE)
class InterpolateOperator(Operator):
    """Logical tensor-boundary traffic, without materialized interpolation weights.

    Trilinear uses the upper bound of eight multiplies and seven additions per
    output value. Edge coincidences and index/coordinate-generation costs are
    not modeled; align_corners is retained as part of the operation signature.
    Nearest-exact performs a copy without floating-point arithmetic. Actual HBM
    rereads and cache reuse require calibration, as with other library operators.
    """

    COMPUTE_UNIT = "vector"
    PARAM_CLASS = InterpolateParams

    def parse_args(self, **kwargs: Any) -> None:
        self.params.validate()
        self.input_shape, self.output_shape = tuple(self.params.input_shape), tuple(self.params.output_shape)
        self.mode, self.align_corners = self.params.mode, self.params.align_corners
        self.input_dtype = self.params.input_dtype or self.compute_dtype
        self.output_dtype = self.params.output_dtype or self.compute_dtype
        if self.input_dtype not in DTYPE_INFO or self.output_dtype not in DTYPE_INFO:
            raise ValueError("unsupported interpolation boundary dtype")
        self._input_size = get_dtype_size(self.input_dtype)
        self._output_size = get_dtype_size(self.output_dtype)
        self.shape = f"{self.input_shape} -> {self.output_shape}; mode={self.mode}; align_corners={self.align_corners}"
        self.assumption = "logical input/output boundary; trilinear 8-neighbor arithmetic upper bound; indexing and HBM rereads excluded"

    def get_flops(self) -> float:
        return float(15 * math.prod(self.output_shape)) if self.mode == "trilinear" else 0.0

    def get_memory_details(self) -> dict[str, int]:
        return self.memory_details(activation=math.prod(self.input_shape) * self._input_size
                                   + math.prod(self.output_shape) * self._output_size)
