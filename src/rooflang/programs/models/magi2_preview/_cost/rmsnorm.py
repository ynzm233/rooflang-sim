"""RMSNorm operator."""

from dataclasses import dataclass, field
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry
from .core import get_dtype_size

_RMS_FLOPS_PER_ELEMENT = 6  # square(1) + mean(2) + add(1) + rsqrt(1) + mul(1) = 6


@dataclass
class RmsNormParams(OperatorParams):
    """RMSNorm parameters."""
    batch_seq: int = field(default=1, metadata={"min": 1, "desc": "批次大小 (B)"})
    dim: int = field(default=768, metadata={"min": 1, "desc": "Feature dimension (D)"})
    affine: bool = True
    weight_banks: int = field(default=1, metadata={"min": 1})
    input_dtype: str | None = None
    output_dtype: str | None = None
    shared_weight: bool = False

@Registry.register_operator("rmsnorm", op_type=OpType.COMPUTE)
class RmsNormOperator(Operator):
    """
    RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma
    """

    COMPUTE_UNIT = "vector"
    PARAM_CLASS = RmsNormParams

    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.batch_seq = params.batch_seq if params is not None else kwargs.get("batch_seq", 0)
        self.dim = params.dim if params is not None else kwargs.get("dim", 0)
        self.affine = params.affine
        self.weight_banks = params.weight_banks
        self.shared_weight = params.shared_weight
        self.input_dtype = params.input_dtype or self.compute_dtype
        self.output_dtype = params.output_dtype or self.compute_dtype
        get_dtype_size(self.input_dtype)
        get_dtype_size(self.output_dtype)

    @property
    def _total_elements(self) -> int:
        return self.batch_seq * self.dim

    def get_flops(self) -> float:
        # Preserve the established affine 6RD convention; without gamma omit
        # its per-element scaling. Reduction precision is compute_dtype.
        return (_RMS_FLOPS_PER_ELEMENT - int(not self.affine)) * self._total_elements

    def get_memory_details(self) -> dict[str, int]:
        """
        - input x: [batch, seq, hidden]
        - output y: [batch, seq, hidden]
        - weight gamma: [hidden]
        """
        w_sz = self.quant_config.get_weight_size()

        # Unspecified boundaries follow any engine precision override, exactly
        # as the original RMSNorm did. Explicit source boundaries remain fixed.
        input_size = (get_dtype_size(self.params.input_dtype) if self.params.input_dtype
                      else self.quant_config.get_compute_size())
        output_size = (get_dtype_size(self.params.output_dtype) if self.params.output_dtype
                       else self.quant_config.get_compute_size())
        activation_bytes = self._total_elements * (input_size + output_size)

        weight_bytes = self.dim * self.weight_banks * w_sz if self.affine else 0

        result = self.memory_details(
            activation=activation_bytes,
            weight=weight_bytes,
        )
        if self.shared_weight:
            result["weight_saved"] = 0
        return result
