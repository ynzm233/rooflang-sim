"""gelu operator."""

from dataclasses import dataclass, field
from typing import Any

from rooflang.programs.models.magi2_preview._cost.core import Operator, OperatorParams, OpType, Registry

# https://huggingface.co/papers/1606.08415
_GELU_FLOPS_PER_ELEMENT = 8.0

@dataclass
class GeluParams(OperatorParams):
    """gelu/Swish activation parameters."""
    batch_seq: int = field(default=1, metadata={"min": 1, "desc": "批次大小 (B)"})
    dim: int = field(default=768, metadata={"min": 1, "desc": "Feature dimension (D)"})

@Registry.register_operator("gelu", op_type=OpType.COMPUTE)
class GeluOperator(Operator):
    """gelu: functools.partial(nn.functional.gelu, approximate="tanh")"""
    
    COMPUTE_UNIT = "vector"
    PARAM_CLASS = GeluParams
    
    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.batch_seq = params.batch_seq if params is not None else kwargs.get("batch_seq", 0)
        self.dim = params.dim if params is not None else kwargs.get("dim", 0)
    
    def get_flops(self) -> float:
        return _GELU_FLOPS_PER_ELEMENT * self.batch_seq * self.dim

    def get_memory_details(self) -> dict[str, int]:
        a_sz = self.quant_config.get_compute_size()
        return self.memory_details(
            activation=2 * self.batch_seq * self.dim * a_sz,  # input + output
        )