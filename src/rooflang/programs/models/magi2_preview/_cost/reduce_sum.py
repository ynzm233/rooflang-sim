"""Reduce Sum operator."""

from dataclasses import dataclass, field
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry


@dataclass
class ReduceSumParams(OperatorParams):
    """Reduce Sum parameters."""
    batch_size: int = field(default=1, metadata={"min": 1, "desc": "批次大小 (B)"})
    seq_len: int = field(default=1, metadata={"min": 1, "desc": "序列长度 (S)"})
    hc_mult: int = field(default=4, metadata={"min": 1, "desc": "HC multiplicity"})
    hidden_size: int = field(default=7168, metadata={"min": 1, "desc": "Hidden dimension"})
    normalize: bool = field(default=False, metadata={"desc": "Divide each reduced value by hc_mult in the same kernel"})


@Registry.register_operator("reduce_sum", op_type=OpType.COMPUTE)
class ReduceSumOperator(Operator):
    """
    Reduce Sum operator: 对多路 hidden states 求和
    y = sum_i x_i, axis=2
    Optional normalization computes the group mean in the same read/write boundary.
    """

    COMPUTE_UNIT = "vector"
    PARAM_CLASS = ReduceSumParams

    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.batch_size = params.batch_size if params is not None else kwargs.get("batch_size", 1)
        self.seq_len = params.seq_len if params is not None else kwargs.get("seq_len", 1)
        self.hc_mult = params.hc_mult if params is not None else kwargs.get("hc_mult", 4)
        self.hidden_size = params.hidden_size if params is not None else kwargs.get("hidden_size", 7168)
        self.normalize = params.normalize if params is not None else kwargs.get("normalize", False)

    @property
    def _num_output_elements(self) -> int:
        return self.batch_size * self.seq_len * self.hidden_size

    @property
    def _num_input_elements(self) -> int:
        return self.batch_size * self.seq_len * self.hc_mult * self.hidden_size

    def get_flops(self) -> float:
        # 每个输出元素需要 (hc_mult - 1) 次加法
        return self._num_output_elements * (self.hc_mult - 1 + int(self.normalize))

    def get_memory_details(self) -> dict[str, int]:
        a_sz = self.quant_config.get_compute_size()
        input_bytes = self._num_input_elements * a_sz
        output_bytes = self._num_output_elements * a_sz
        activation_bytes = input_bytes + output_bytes
        return self.memory_details(
            activation=activation_bytes,
        )
