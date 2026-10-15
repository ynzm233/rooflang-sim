"""Transposed 1D convolution, counted at its complete operator boundary."""

from dataclasses import dataclass, field
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry


@dataclass
class ConvTranspose1dParams(OperatorParams):
    batch: int = field(default=1, metadata={"min": 1})
    c_in: int = field(default=1, metadata={"min": 1})
    L_in: int = field(default=1, metadata={"min": 1})
    c_out: int = field(default=1, metadata={"min": 1})
    kernel_size: int = field(default=1, metadata={"min": 1})
    stride: int = field(default=1, metadata={"min": 1})
    pad: int = field(default=0, metadata={"min": 0})
    dilation: int = field(default=1, metadata={"min": 1})
    output_padding: int = field(default=0, metadata={"min": 0})
    group: int = field(default=1, metadata={"min": 1})
    has_bias: bool = False

    def validate(self) -> None:
        super().validate()
        if self.c_in % self.group or self.c_out % self.group:
            raise ValueError("ConvTranspose1d channels must be divisible by group")
        if self.output_padding >= max(self.stride, self.dilation):
            raise ValueError("output_padding must be smaller than stride or dilation")


@Registry.register_operator("conv_transpose1d", op_type=OpType.COMPUTE)
class ConvTranspose1dOperator(Operator):
    """Input-kernel MAC upper bound, including products cropped at the edges.

    This follows the published MAGI audio convolution convention. It is not an
    output-length Conv1d proxy or a prediction of a particular cuDNN algorithm.
    Bias follows the existing Matmul/Conv cube accounting convention.
    """

    COMPUTE_UNIT = "cube"
    PARAM_CLASS = ConvTranspose1dParams

    def parse_args(self, **kwargs: Any) -> None:
        self.params.validate()
        for name in self.params.__dataclass_fields__:
            setattr(self, name, getattr(self.params, name))
        self.L_out = ((self.L_in - 1) * self.stride - 2 * self.pad
                      + self.dilation * (self.kernel_size - 1)
                      + self.output_padding + 1)
        if self.L_out <= 0:
            raise ValueError("ConvTranspose1d output length must be positive")
        self._input_elements = self.batch * self.c_in * self.L_in
        self._output_elements = self.batch * self.c_out * self.L_out
        self._weight_elements = self.c_in * (self.c_out // self.group) * self.kernel_size
        self.shape = (f"[{self.batch},{self.c_in},{self.L_in}] -> "
                      f"[{self.batch},{self.c_out},{self.L_out}]; kernel={self.kernel_size}; "
                      f"stride={self.stride}; dilation={self.dilation}")
        self.assumption = "input-kernel MAC upper bound including edge-cropped products"

    def get_flops(self) -> float:
        return float(2 * self.batch * self.L_in * self._weight_elements
                     + (self._output_elements if self.has_bias else 0))

    def get_memory_details(self) -> dict[str, int]:
        return self.memory_details(
            activation=(self._input_elements + self._output_elements)
            * self.quant_config.get_compute_size(),
            weight=(self._weight_elements + (self.c_out if self.has_bias else 0))
            * self.quant_config.get_weight_size(),
        )
