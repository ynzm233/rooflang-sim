"""LayerNorm operator."""

from dataclasses import dataclass, field
from typing import Any

from rooflang.programs.models.magi2_preview._cost.core import Operator, OperatorParams, OpType, Registry

_LAYERNORM_FLOPS_PER_ELEMENT = 8


@dataclass
class LayerNormParams(OperatorParams):
    """LayerNorm parameters."""
    batch_seq: int = field(default=1, metadata={"min": 1, "desc": "批次大小 (B)"})
    dim: int = field(default=768, metadata={"min": 1, "desc": "Feature dimension (D)"})

@Registry.register_operator("layernorm", op_type=OpType.COMPUTE)
class LayerNormOperator(Operator):
    
    COMPUTE_UNIT = "vector"
    PARAM_CLASS = LayerNormParams
    
    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.batch_seq = params.batch_seq if params is not None else kwargs.get("batch_seq", 0)
        self.dim = params.dim if params is not None else kwargs.get("dim", 0)

    @property
    def _total_elements(self) -> int:
        return self.batch_seq * self.dim

    def get_flops(self) -> float:
        return _LAYERNORM_FLOPS_PER_ELEMENT * self._total_elements
    
    def get_memory_details(self) -> dict[str, int]:
        """
        - input x: [batch, seq, hidden]
        - output y: [batch, seq, hidden]
        - weight gamma: [hidden]
        """
        a_sz = self.quant_config.get_compute_size()
        w_sz = self.quant_config.get_weight_size()
        
        activation_bytes = 2 * self._total_elements * a_sz
        
        weight_bytes = 2 * self.dim * w_sz
        
        return self.memory_details(
            activation=activation_bytes,
            weight=weight_bytes,
        )