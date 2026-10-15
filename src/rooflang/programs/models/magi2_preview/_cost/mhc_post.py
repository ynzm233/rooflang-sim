"""MHC Post operator for mHC."""

from dataclasses import dataclass, field
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry


@dataclass
class MHCPostParams(OperatorParams):
    """MHC Post parameters."""
    batch_size: int = field(default=1, metadata={"min": 1, "desc": "批次大小 (B)"})
    seq_len: int = field(default=1, metadata={"min": 1, "desc": "序列长度 (S)"})
    hc_mult: int = field(default=4, metadata={"min": 1, "desc": "HC multiplicity"})
    hidden_size: int = field(default=7168, metadata={"min": 1, "desc": "Hidden dimension"})


@Registry.register_operator("mhc_post", op_type=OpType.COMPUTE)
class MHCPostOperator(Operator):
    """
    MHC Post-mapping: 合并多路结果 + 残差融合
    y = post * x @ comb^T + residual
    """

    COMPUTE_UNIT = "cube"
    PARAM_CLASS = MHCPostParams

    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.batch_size = params.batch_size if params is not None else kwargs.get("batch_size", 1)
        self.seq_len = params.seq_len if params is not None else kwargs.get("seq_len", 1)
        self.hc_mult = params.hc_mult if params is not None else kwargs.get("hc_mult", 4)
        self.hidden_size = params.hidden_size if params is not None else kwargs.get("hidden_size", 7168)

    @property
    def _num_elements(self) -> int:
        input_elements = self.batch_size * self.seq_len * self.hc_mult * self.hidden_size + self.batch_size * self.seq_len * self.hidden_size
        # Hpost_mul_attn/moe_output
        output_elements = self.batch_size * self.seq_len * self.hidden_size
        # HRes_mul_x

        #add

        return input_elements + output_elements

    def get_flops(self) -> float:
        # Hyes_mul_x
        hpres_mul_x_flops = self.batch_size * self.seq_len * (2*self.hc_mult*self.hc_mult-self.hc_mult) * self.hidden_size
        # Hpost_mul_attn/moe
        hpost_mul_y_flops = self.batch_size * self.seq_len *self.hc_mult * self.hidden_size
        # add
        add_flops =  self.batch_size * self.seq_len *self.hc_mult * self.hidden_size
        return hpres_mul_x_flops + hpost_mul_y_flops + add_flops

    def get_memory_details(self) -> dict[str, int]:
        a_sz = self.quant_config.get_compute_size()
        activation_bytes = self._num_elements * a_sz  # x, post, comb, residual, output
        return self.memory_details(
            activation=activation_bytes,
        )