"""Conv1d operator."""

from dataclasses import dataclass, field
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry

_CONV1D_FLOPS_FACTOR = 2.0


@dataclass
class Conv1dParams(OperatorParams):
    """Conv1d parameters."""
    batch: int = field(default=1, metadata={"min": 1, "desc": "批次大小"})
    c_in: int = field(default=1, metadata={"min": 1, "desc": "输入通道数"})
    L_in: int = field(default=1, metadata={"min": 1, "desc": "输入序列长度"})
    c_out: int = field(default=1, metadata={"min": 1, "desc": "输出通道数"})
    group: int = field(default=1, metadata={"min": 1, "desc": "分组卷积"})
    kernel_size: int = field(default=1, metadata={"min": 1, "desc": "卷积核大小"})
    pad: int = field(default=0, metadata={"min": 0, "desc": "填充大小"})
    stride: int = field(default=1, metadata={"min": 1, "desc": "步长"})
    dilation: int = field(default=1, metadata={"min": 1, "desc": "卷积核采样间距"})
    has_bias: bool = field(default=False, metadata={"desc": "是否计入输出偏置"})
    cal_kv_cache: bool = field(default=False, metadata={"desc": "是否计算KV缓存"})


@Registry.register_operator("conv1d", op_type=OpType.COMPUTE)
class Conv1dOperator(Operator):
    """
    Conv1d: 1D convolution over an input sequence
    """

    COMPUTE_UNIT = "cube"
    PARAM_CLASS = Conv1dParams

    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.batch = params.batch if params is not None else kwargs.get("batch", 0)
        self.c_in = params.c_in if params is not None else kwargs.get("c_in", 0)
        self.L_in = params.L_in if params is not None else kwargs.get("L_in", 0)
        self.c_out = params.c_out if params is not None else kwargs.get("c_out", 0)
        self.group = params.group if params is not None else kwargs.get("group", 0)
        self.kernel_size = params.kernel_size if params is not None else kwargs.get("kernel_size", 0)
        self.pad = params.pad if params is not None else kwargs.get("pad", 0)
        self.stride = params.stride if params is not None else kwargs.get("stride", 1)
        self.dilation = params.dilation if params is not None else kwargs.get("dilation", 1)
        self.has_bias = params.has_bias if params is not None else kwargs.get("has_bias", False)
        self.cal_kv_cache = params.cal_kv_cache if params is not None else kwargs.get("cal_kv_cache", False)

        self.L_out = (self.L_in + 2 * self.pad - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        self._input_elements = self.batch * self.c_in * self.L_in
        self._output_elements = self.batch * self.c_out * self.L_out
        self._weight_elements = (self.c_in // self.group) * self.c_out * self.kernel_size

    def get_flops(self) -> float:
        # As in MatmulOperator, an optional bias adds one cube FLOP per output.
        return (_CONV1D_FLOPS_FACTOR * self.batch * (self.c_in // self.group)
                * self.c_out * self.kernel_size * self.L_out
                + (self._output_elements if self.has_bias else 0))

    def get_memory_details(self) -> dict[str, int]:
        a_sz = self.quant_config.get_compute_size()
        w_sz = self.quant_config.get_weight_size()

        activation_bytes = self._input_elements * a_sz + self._output_elements * a_sz
        weight_bytes = (self._weight_elements + (self.c_out if self.has_bias else 0)) * w_sz
        kv_cache_bytes = self.batch * self.c_in * self.dilation * (self.kernel_size - 1) * a_sz if self.cal_kv_cache else 0
        return self.memory_details(
            activation=activation_bytes,
            weight=weight_bytes,
            kv_cache=kv_cache_bytes,
        )
