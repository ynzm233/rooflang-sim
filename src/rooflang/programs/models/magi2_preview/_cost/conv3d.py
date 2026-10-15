"""Conv3d operator for 3D convolutions (used in VAE)."""

from dataclasses import dataclass, field
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry

_CONV3D_FLOPS_FACTOR = 2.0


@dataclass
class Conv3dParams(OperatorParams):
    """Conv3d parameters (causal 3d conv)."""
    batch: int = field(default=1, metadata={"min": 1, "desc": "批次大小"})
    c_in: int = field(default=1, metadata={"min": 1, "desc": "输入通道数"})
    c_out: int = field(default=1, metadata={"min": 1, "desc": "输出通道数"})
    d_in: int = field(default=1, metadata={"min": 1, "desc": "输入时间维度 (T)"})
    h_in: int = field(default=1, metadata={"min": 1, "desc": "输入高度 (H)"})
    w_in: int = field(default=1, metadata={"min": 1, "desc": "输入宽度 (W)"})
    kernel_t: int = field(default=3, metadata={"min": 1, "desc": "时间卷积核大小"})
    kernel_h: int = field(default=1, metadata={"min": 1, "desc": "高度卷积核大小"})
    kernel_w: int = field(default=1, metadata={"min": 1, "desc": "宽度卷积核大小"})
    stride_t: int = field(default=1, metadata={"min": 1, "desc": "时间步长"})
    stride_h: int = field(default=1, metadata={"min": 1, "desc": "高度步长"})
    stride_w: int = field(default=1, metadata={"min": 1, "desc": "宽度步长"})
    pad_t: int = field(default=0, metadata={"min": 0, "desc": "时间填充"})
    pad_h: int = field(default=0, metadata={"min": 0, "desc": "高度填充"})
    pad_w: int = field(default=0, metadata={"min": 0, "desc": "宽度填充"})
    groups: int = field(default=1, metadata={"min": 1, "desc": "分组卷积"})
    has_bias: bool = field(default=False, metadata={"desc": "是否计入输出偏置"})

    def __post_init__(self):
        # compute output dimensions
        self.d_out = (self.d_in + 2 * self.pad_t - (self.kernel_t - 1) - 1) // self.stride_t + 1
        self.h_out = (self.h_in + 2 * self.pad_h - (self.kernel_h - 1) - 1) // self.stride_h + 1
        self.w_out = (self.w_in + 2 * self.pad_w - (self.kernel_w - 1) - 1) // self.stride_w + 1


@Registry.register_operator("conv3d", op_type=OpType.COMPUTE)
class Conv3dOperator(Operator):
    """
    Conv3d: 3D convolution over (T, H, W).
    Used in VAE encoder/decoder blocks.
    """

    COMPUTE_UNIT = "cube"
    PARAM_CLASS = Conv3dParams

    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.batch = params.batch if params is not None else kwargs.get("batch", 1)
        self.c_in = params.c_in if params is not None else kwargs.get("c_in", 1)
        self.c_out = params.c_out if params is not None else kwargs.get("c_out", 1)
        self.d_in = params.d_in if params is not None else kwargs.get("d_in", 1)
        self.h_in = params.h_in if params is not None else kwargs.get("h_in", 1)
        self.w_in = params.w_in if params is not None else kwargs.get("w_in", 1)
        self.kernel_t = params.kernel_t if params is not None else kwargs.get("kernel_t", 3)
        self.kernel_h = params.kernel_h if params is not None else kwargs.get("kernel_h", 1)
        self.kernel_w = params.kernel_w if params is not None else kwargs.get("kernel_w", 1)
        self.stride_t = params.stride_t if params is not None else kwargs.get("stride_t", 1)
        self.stride_h = params.stride_h if params is not None else kwargs.get("stride_h", 1)
        self.stride_w = params.stride_w if params is not None else kwargs.get("stride_w", 1)
        self.pad_t = params.pad_t if params is not None else kwargs.get("pad_t", 0)
        self.pad_h = params.pad_h if params is not None else kwargs.get("pad_h", 0)
        self.pad_w = params.pad_w if params is not None else kwargs.get("pad_w", 0)
        self.groups = params.groups if params is not None else kwargs.get("groups", 1)
        self.has_bias = params.has_bias if params is not None else kwargs.get("has_bias", False)

        # output dimensions
        self.d_out = (self.d_in + 2 * self.pad_t - (self.kernel_t - 1) - 1) // self.stride_t + 1
        self.h_out = (self.h_in + 2 * self.pad_h - (self.kernel_h - 1) - 1) // self.stride_h + 1
        self.w_out = (self.w_in + 2 * self.pad_w - (self.kernel_w - 1) - 1) // self.stride_w + 1

        self._input_elements = self.batch * self.c_in * self.d_in * self.h_in * self.w_in
        self._output_elements = self.batch * self.c_out * self.d_out * self.h_out * self.w_out
        self._weight_elements = (self.c_in // self.groups) * self.c_out * self.kernel_t * self.kernel_h * self.kernel_w

    def get_flops(self) -> float:
        return (_CONV3D_FLOPS_FACTOR
                * self.batch
                * (self.c_in // self.groups)
                * self.c_out
                * self.kernel_t * self.kernel_h * self.kernel_w
                * self.d_out * self.h_out * self.w_out
                + (self._output_elements if self.has_bias else 0))

    def get_memory_details(self) -> dict[str, int]:
        a_sz = self.quant_config.get_compute_size()
        w_sz = self.quant_config.get_weight_size()

        activation_bytes = self._input_elements * a_sz + self._output_elements * a_sz
        weight_bytes = (self._weight_elements + (self.c_out if self.has_bias else 0)) * w_sz
        return self.memory_details(
            activation=activation_bytes,
            weight=weight_bytes,
        )
