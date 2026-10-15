"""Shape-based feature transforms with explicit mathematical/storage boundaries.

Special functions count as scalar operations here, except sigmoid, which uses
the existing SigmoidOperator convention. These remain uncalibrated rooflines.
"""

from dataclasses import dataclass, field
import math

from .core import Operator, OperatorParams, OpType, Registry
from .core import get_dtype_size
from .sigmoid import SigmoidOperator


@dataclass
class WeightNormParams(OperatorParams):
    weight_shape: list[int] = field(default_factory=list)
    dim: int = 0


@Registry.register_operator("weight_norm", op_type=OpType.COMPUTE)
class WeightNormOperator(Operator):
    """Materialize g*v/||v||, with one scale for each slice along dim."""

    PARAM_CLASS = WeightNormParams

    def parse_args(self, **kwargs):
        self.weight_shape = tuple(self.params.weight_shape)
        if not self.weight_shape or any(type(n) is not int or n < 1 for n in self.weight_shape):
            raise ValueError("weight_shape must contain positive dimensions")
        if not 0 <= self.params.dim < len(self.weight_shape):
            raise ValueError("weight_norm dim outside weight_shape")
        self.elements = math.prod(self.weight_shape)
        self.groups = self.weight_shape[self.params.dim]

    def get_flops(self):
        # Square + reduction + sqrt + g/norm + elementwise scaling.
        return 3 * self.elements + self.groups

    def get_memory_details(self):
        size = self.quant_config.get_compute_size()
        # v/materialized kernels belong to the convolution residency inventory.
        return self.memory_details(activation=2 * self.elements * size,
                                   weight=self.groups * self.quant_config.get_weight_size())


@dataclass
class SigmoidGateParams(OperatorParams):
    batch_seq: int = field(default=1, metadata={"min": 1})
    num_heads: int = field(default=1, metadata={"min": 1})
    head_dim: int = field(default=1, metadata={"min": 1})
    input_dtype: str | None = None
    output_dtype: str | None = None


@Registry.register_operator("sigmoid_gate", op_type=OpType.COMPUTE)
class SigmoidGateOperator(Operator):
    """Broadcast sigmoid(gate) over each attention head's output vector."""

    PARAM_CLASS = SigmoidGateParams

    def parse_args(self, **kwargs):
        p = self.params
        self.gates = p.batch_seq * p.num_heads
        self.elements = self.gates * p.head_dim
        self.input_dtype = p.input_dtype or self.compute_dtype
        self.output_dtype = p.output_dtype or self.compute_dtype

    def get_flops(self):
        return SigmoidOperator(batch_seq=self.gates, dim=1).get_flops() + self.elements

    def get_memory_details(self):
        return self.memory_details(activation=(
            self.gates * self.quant_config.get_compute_size() + self.elements * (
                get_dtype_size(self.input_dtype) + get_dtype_size(self.output_dtype))))


@dataclass
class FourierFeaturesParams(OperatorParams):
    batch_seq: int = field(default=1, metadata={"min": 1})
    num_bands: int = field(default=1, metadata={"min": 1})
    coordinate_dims: int = field(default=3, metadata={"min": 1})


@Registry.register_operator("fourier_features", op_type=OpType.COMPUTE)
class FourierFeaturesOperator(Operator):
    """Scale/center coordinates, project to fixed bands, concatenate sin/cos.

    MAGI ElementWiseFourierEmbed consumes xyz, sizes and reference sizes. The
    regular-path scalar formula omits validation/exception checks and band init.
    """

    PARAM_CLASS = FourierFeaturesParams

    def parse_args(self, **kwargs):
        p = self.params
        self.coords = p.batch_seq * p.coordinate_dims
        self.projected = self.coords * p.num_bands

    def get_flops(self):
        # (refs-1)/(sizes-1), center, subtract: six per axis;
        # coords*scales is evaluated before broadcast to bands: one multiply
        # per coordinate, then a band multiply plus sin/cos per projection.
        return 7 * self.coords + 3 * self.projected

    def get_memory_details(self):
        return self.memory_details(activation=(3 * self.coords + 2 * self.projected) *
                                   self.quant_config.get_compute_size())


@dataclass
class ReluParams(OperatorParams):
    element_count: int = field(default=1, metadata={"min": 1})


@Registry.register_operator("relu", op_type=OpType.COMPUTE)
class ReluOperator(Operator):
    PARAM_CLASS = ReluParams

    def parse_args(self, **kwargs):
        self.element_count = self.params.element_count

    def get_flops(self):
        return self.element_count

    def get_memory_details(self):
        return self.memory_details(activation=2 * self.element_count * self.quant_config.get_compute_size())
