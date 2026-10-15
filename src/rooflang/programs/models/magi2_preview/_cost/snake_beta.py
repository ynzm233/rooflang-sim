"""Periodic SnakeBeta activation, including learned alpha/beta storage."""

from dataclasses import dataclass, field
import math
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry


@dataclass
class SnakeBetaParams(OperatorParams):
    batch: int = field(default=1, metadata={"min": 1})
    channels: int = field(default=1, metadata={"min": 1})
    length: int = field(default=1, metadata={"min": 1})
    logscale: bool = True
    sin_flops: float = field(default=1.0, metadata={"min": 0})
    exp_flops: float = field(default=1.0, metadata={"min": 0})


@Registry.register_operator("snake_beta", op_type=OpType.COMPUTE)
class SnakeBetaOperator(Operator):
    """x + sin(alpha*x)^2 / (beta+eps).

    Count multiply/square/multiply/add per element. Beta epsilon and reciprocal,
    and optional exponentiation of log alpha/beta, occur per channel rather
    than once per sample. Squaring is a multiplication, not a generic pow call.
    """

    COMPUTE_UNIT = "vector"
    PARAM_CLASS = SnakeBetaParams

    def parse_args(self, **kwargs: Any) -> None:
        self.params.validate()
        for name in self.params.__dataclass_fields__:
            setattr(self, name, getattr(self.params, name))
        if not all(math.isfinite(x) and x >= 0 for x in (self.sin_flops, self.exp_flops)):
            raise ValueError("SnakeBeta special-function costs must be finite and nonnegative")
        self.shape = f"[{self.batch},{self.channels},{self.length}]"
        self.assumption = "one activation boundary; channel constants computed once; sin/exp costs are proxies"

    def get_flops(self) -> float:
        per_channel = 2 + (2 * self.exp_flops if self.logscale else 0)
        return float(self.batch * self.channels * self.length * (4 + self.sin_flops)
                     + self.channels * per_channel)

    def get_memory_details(self) -> dict[str, int]:
        return self.memory_details(
            activation=2 * self.batch * self.channels * self.length * self.quant_config.get_compute_size(),
            weight=2 * self.channels * self.quant_config.get_weight_size(),
        )
