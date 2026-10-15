"""MAGI-style mHC call boundaries, separate from the existing CANN pre operator.

The existing MHCPreOperator includes norm, Phi and a different exchange model;
using it after explicit norm/Phi would count those operations twice. This
operator contains only the named coefficient/mixing phase. Legacy mHC operators
and their default formulas are unchanged.
"""

from dataclasses import dataclass, field
import math
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry
from .core import DTYPE_INFO, get_dtype_size
from .mhc_post import MHCPostOperator
from .swiglu7 import DEFAULT_SIGMOID_FLOPS


@dataclass
class MHCParams(OperatorParams):
    phase: str = field(default="pre", metadata={"choices": ["coefficients", "sinkhorn", "pre", "post"]})
    batch_seq: int = field(default=1, metadata={"min": 1})
    hidden_size: int = field(default=1, metadata={"min": 1})
    streams: int = field(default=4, metadata={"min": 1})
    iterations: int = field(default=20, metadata={"min": 1})
    sigmoid_flops: float = field(default=DEFAULT_SIGMOID_FLOPS, metadata={"min": 0})
    exp_flops: float = field(default=1.0, metadata={"min": 0})
    input_dtype: str | None = None
    output_dtype: str | None = None
    coefficient_dtype: str | None = None


@Registry.register_operator("mhc", op_type=OpType.COMPUTE)
class MHCOperator(Operator):
    """One semantic phase of stream coefficients, Sinkhorn or stream mixing.

    ``coefficients`` covers both affine sigmoid pre/post vectors (including the
    scale multiplications), and ``sinkhorn`` covers the residual affine matrix,
    stable exp initialization, then both normalization axes on every iteration.
    A reduction of n values costs n-1 additions, an epsilon add costs one and
    each division costs one: the two axes cost 4*n*n operations per iteration.

    Pre/post stream reductions execute in the source's scalar/vector kernels,
    so use vector throughput. Post uses the existing MHCPost mathematical
    count; its source boundary additionally includes all coefficient reads and
    the full multi-stream output. Phi matmul and RMSNorm are separate operators.
    """

    COMPUTE_UNIT = "vector"
    PARAM_CLASS = MHCParams

    def parse_args(self, **kwargs: Any) -> None:
        self.params.validate()
        for name in self.params.__dataclass_fields__:
            setattr(self, name, getattr(self.params, name))
        if not all(math.isfinite(x) and x >= 0 for x in (self.sigmoid_flops, self.exp_flops)):
            raise ValueError("mHC special-function costs must be finite and nonnegative")
        self.input_dtype = self.input_dtype or self.compute_dtype
        self.output_dtype = self.output_dtype or self.compute_dtype
        self.coefficient_dtype = self.coefficient_dtype or self.input_dtype
        if any(dtype not in DTYPE_INFO for dtype in (self.input_dtype, self.output_dtype, self.coefficient_dtype)):
            raise ValueError("unsupported mHC boundary dtype")
        self._input_size = get_dtype_size(self.input_dtype)
        self._output_size = get_dtype_size(self.output_dtype)
        self._coefficient_size = get_dtype_size(self.coefficient_dtype)
        self.shape = (f"phase={self.phase}; rows={self.batch_seq}; streams={self.streams}; "
                      f"hidden={self.hidden_size}; iterations={self.iterations}")
        self.assumption = "source vector kernels; complete phase boundary; sigmoid/exp costs are explicit proxies"

    def get_flops(self) -> float:
        rows, n, dim = self.batch_seq, self.streams, self.hidden_size
        if self.phase == "coefficients":
            return float(rows * 2 * n * (4 + self.sigmoid_flops))
        if self.phase == "sinkhorn":
            # Affine: mul2+add1, stable max: n²-1 comparisons, subtraction,
            # exponentiation, and the two normalization axes each iteration.
            return float(rows * (n*n * (4 + self.exp_flops) + (n*n - 1)
                                 + self.iterations * 4 * n*n))
        if self.phase == "pre":
            return float(rows * dim * (2 * n - 1))
        return MHCPostOperator(batch_size=1, seq_len=rows, hc_mult=n,
                               hidden_size=dim, quant_config=self.quant_config).get_flops()

    def get_memory_details(self) -> dict[str, int]:
        rows, n, dim = self.batch_seq, self.streams, self.hidden_size
        weight_elements = 0
        if self.phase == "coefficients":
            activation = rows * 2*n * (self._input_size + self._output_size)
            weight_elements = 2 + 2*n  # Two alpha scalars, pre/post bias vectors.
        elif self.phase == "sinkhorn":
            activation = rows * n*n * (self._input_size + self._output_size)
            weight_elements = 1 + n*n  # Residual alpha and residual bias.
        elif self.phase == "pre":
            activation = rows * (n*dim*self._input_size + n*self._coefficient_size
                                 + dim*self._output_size)
        else:
            activation = rows * ((n*dim+dim)*self._input_size
                                 + (n+n*n)*self._coefficient_size
                                 + n*dim*self._output_size)
        return self.memory_details(activation=activation,
                                   weight=weight_elements*self.quant_config.get_weight_size())
