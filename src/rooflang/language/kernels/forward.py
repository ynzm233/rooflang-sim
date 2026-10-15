# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Forward-pass Kernel subclasses for the model-roofline op enumerators.

Each class takes shape + dtype arguments and exposes roofline metrics
(flops, input_bytes, weight_bytes, output_bytes) as @property methods.

The "bytes" definition is per-kernel: GEMM treats every input as freshly
read from HBM (no reuse); attention assumes flash-style SMEM reuse of K/V
tiles (no S² in HBM). Each docstring states its own reuse model.
"""

from fractions import Fraction

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.utils import dtype_bytes, gemm_scale_bytes


class Nop(Kernel):
    """Zero-cost dependency kernel with arbitrary tensor ports."""

    _requires_placement = False

    @property
    def flops(self) -> float:
        return 0.0

    @property
    def input_bytes(self) -> float:
        return 0.0

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return 0.0

    @property
    def transferred_bytes(self) -> float:
        return 0.0


class ReadInput(Kernel):
    """Host-to-device transfer: read token indices from CPU memory to GPU.

    flops = 0 (pure memcpy).
    bytes:
        input_bytes  = n_elements · sizeof(dtype)   (from CPU DRAM)
        output_bytes = n_elements · sizeof(dtype)   (to GPU HBM)
    """

    def __init__(self, n_elements: int, dtype: str = "int32"):
        self.n_elements = n_elements
        self.dtype_ = dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return 0.0

    @property
    def input_bytes(self) -> float:
        return self.n_elements * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.n_elements * dtype_bytes(self.dtype_)


class Slice(Kernel):
    """Materialize a contiguous slice into a compact output allocation.

    The graph builder supplies the full source tensor as the input and the
    selected region as the output.  The generic Kernel byte accounting reads
    the input tensor and writes the compact output tensor.  Unlike a view,
    the output does not alias or pin the source allocation.
    """

    def __init__(self):
        super().__init__()


class Embedding(Kernel):
    """Token embedding lookup: gather M rows from a V×D table.

    flops = 0 (pure gather, no arithmetic).
    bytes:
        input_bytes  = M · sizeof(idx_dtype)        (token indices)
        weight_bytes = V · D · sizeof(w_dtype)      (full embedding table in HBM)
        output_bytes = M · D · sizeof(out_dtype)
    """

    def __init__(self, M: int, V: int, D: int,
                 w_dtype: str = "bf16", idx_dtype: str = "int32",
                 out_dtype: str = "bf16"):
        self.M, self.V, self.D = M, V, D
        self.w_dtype = w_dtype
        self.idx_dtype = idx_dtype
        self.out_dtype = out_dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return 0.0

    @property
    def input_bytes(self) -> float:
        return self.M * dtype_bytes(self.idx_dtype)

    @property
    def weight_bytes(self) -> float:
        return self.V * self.D * dtype_bytes(self.w_dtype)

    @property
    def output_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.out_dtype)


class ElementwiseOp(Kernel):
    """Element-wise binary op: y = op(a, b).

    Supported ops:
      "add": y = a + b
      "mul": y = a * b
      "sigmoid_mul": y = a * sigmoid(b)

    flops = M·D for add/mul, 5·M·D for a sigmoid followed by multiply.
    bytes:
        input_bytes  = 2·M·D · sizeof(dtype)  (two input tensors)
        weight_bytes = 0
        output_bytes = M·D · sizeof(dtype)
    """

    def __init__(self, M: int, D: int, dtype: str = "bf16",
                 op: str = "add"):
        self.M, self.D, self.dtype_ = M, D, dtype
        self.op = op
        super().__init__()

    @property
    def flops(self) -> float:
        ops_per_element = 5 if self.op == "sigmoid_mul" else 1
        return float(ops_per_element * self.M * self.D)

    @property
    def input_bytes(self) -> float:
        return 2.0 * self.M * self.D * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.dtype_)


class Gemm(Kernel):
    """Dense GEMM C(M,N) = A(M,K) · B(K,N).

    flops = 2·M·N·K (fused MAC counted as 2 per output element).
    bytes (HBM, no reuse):
        input_bytes  = M·K · sizeof(a_dtype)
        weight_bytes = K·N · sizeof(w_dtype) + scale bytes
        output_bytes = M·N · sizeof(out_dtype)
    """

    def __init__(self, M: int, N: int, K: int,
                 w_dtype: str, a_dtype: str, out_dtype: str = "bf16"):
        self.M, self.N, self.K = M, N, K
        self.w_dtype, self.a_dtype, self.out_dtype = w_dtype, a_dtype, out_dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return 2.0 * self.M * self.N * self.K

    @property
    def input_bytes(self) -> float:
        return self.M * self.K * dtype_bytes(self.a_dtype)

    @property
    def weight_bytes(self) -> float:
        return (self.K * self.N * dtype_bytes(self.w_dtype)
                + gemm_scale_bytes(self.N, self.K, self.w_dtype))

    @property
    def output_bytes(self) -> float:
        return self.M * self.N * dtype_bytes(self.out_dtype)


class RMSNorm(Kernel):
    """RMSNorm:  y = x · rsqrt(mean(x²) + eps) · gamma.

    flops = 4·M·D (square + reduce + mul-by-rsqrt + mul-by-gamma).
    bytes (fused single-pass):
        input_bytes  = M·D · sizeof(dtype)
        weight_bytes = D · sizeof(dtype)   (gamma, broadcast)
        output_bytes = M·D · sizeof(dtype)
    """

    def __init__(self, M: int, D: int, dtype: str = "bf16"):
        self.M, self.D, self.dtype_ = M, D, dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return 4.0 * self.M * self.D

    @property
    def input_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return self.D * dtype_bytes(self.dtype_)

    @property
    def output_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.dtype_)


class PartialRMSNorm(Kernel):
    """RMSNorm a prefix while forwarding the remaining features unchanged."""

    def __init__(self, M: int, input_dim: int, norm_dim: int,
                 dtype: str = "bf16"):
        if not 0 < norm_dim <= input_dim:
            raise ValueError(
                f"norm_dim must be in [1, input_dim], got {norm_dim}")
        self.M, self.input_dim = M, input_dim
        self.norm_dim, self.dtype_ = norm_dim, dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return 4.0 * self.M * self.norm_dim

    @property
    def input_bytes(self) -> float:
        return self.M * self.input_dim * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return self.norm_dim * dtype_bytes(self.dtype_)

    @property
    def output_bytes(self) -> float:
        return self.M * self.input_dim * dtype_bytes(self.dtype_)


class LayerNorm(Kernel):
    """LayerNorm:  y = (x-mean)·rsqrt(var+eps)·gamma + beta.

    flops = 7·M·D (mean + sub + sq + var + mul-invstd + mul-gamma + add-beta).
    bytes (fused single-pass):
        input_bytes  = M·D · sizeof(dtype)
        weight_bytes = 2·D · sizeof(dtype)  (gamma + beta)
        output_bytes = M·D · sizeof(dtype)
    """

    def __init__(self, M: int, D: int, dtype: str = "bf16"):
        self.M, self.D, self.dtype_ = M, D, dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return 7.0 * self.M * self.D

    @property
    def input_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return 2 * self.D * dtype_bytes(self.dtype_)

    @property
    def output_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.dtype_)


class AttnRes(Kernel):
    """Kimi AttnRes weighted residual aggregation.

    ``R`` is the number of saved block residuals.  The current prefix sum is
    appended internally, so every token scores and combines ``R + 1`` hidden
    vectors.  Kimi's implementation casts the aggregation path to FP32.
    """

    def __init__(self, B: int, S: int, D: int, R: int,
                 dtype: str = "bf16", compute_dtype: str = "fp32"):
        self.B, self.S, self.D, self.R = B, S, D, R
        self.dtype_ = compute_dtype
        self.storage_dtype = dtype
        super().__init__()

    @property
    def flops(self) -> float:
        candidates = self.R + 1
        tokens = self.B * self.S
        # RMS normalization + score dot + weighted reduction, plus softmax.
        return float(
            7 * tokens * candidates * self.D
            + 5 * tokens * candidates
            + self.D
        )

    @property
    def input_bytes(self) -> float:
        return (self.B * self.S * (self.R + 1) * self.D
                * dtype_bytes(self.storage_dtype))

    @property
    def weight_bytes(self) -> float:
        return 2 * self.D * dtype_bytes(self.storage_dtype)

    @property
    def output_bytes(self) -> float:
        return (self.B * self.S * self.D
                * dtype_bytes(self.storage_dtype))


class Attn(Kernel):
    """Flash-style multi-head attention forward (no S² in HBM).

    flops = 4·B·H·S_q·S_kv·Hd (×0.5 for a triangular causal matrix).
    bytes (flash-tiled — K/V reused in SMEM):
        input_bytes  = (B·H·S_q·Hd + 2·B·H_kv·S_kv·Hd) · sizeof(dtype)
        weight_bytes = 0
        output_bytes = B·H·S_q·Hd · sizeof(dtype)
    """

    def __init__(self, B: int, H: int, H_kv: int,
                 S_q: int, S_kv: int, Hd: int,
                 dtype: str = "bf16", causal: bool = False,
                 triangular: bool | None = None):
        self.B, self.H, self.H_kv = B, H, H_kv
        self.S_q, self.S_kv, self.Hd = S_q, S_kv, Hd
        self.dtype_, self.causal = dtype, causal
        self.triangular = (causal and S_q == S_kv
                           if triangular is None else triangular)
        super().__init__()

    @property
    def flops(self) -> float:
        f = 4.0 * self.B * self.H * self.S_q * self.S_kv * self.Hd
        if self.triangular:
            f *= 0.5
        return f

    @property
    def input_bytes(self) -> float:
        b = dtype_bytes(self.dtype_)
        return (self.B * self.H * self.S_q * self.Hd
                + 2 * self.B * self.H_kv * self.S_kv * self.Hd) * b

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.B * self.H * self.S_q * self.Hd * dtype_bytes(self.dtype_)


class StridedGemm(Kernel):
    """Gemm-like kernel where I/O shapes differ from standard M*K / M*N.

    Used for grouped linears (input != M*K), fused gated projections
    (output != M*N), or strided writes (output rows != M).
    Flops still follow 2*M*N*K.

    bytes:
        input_bytes  = in_elems · sizeof(a_dtype)
        weight_bytes = K·N · sizeof(w_dtype) + scale
        output_bytes = out_elems · sizeof(out_dtype)
    """

    def __init__(self, M: int, N: int, K: int,
                 w_dtype: str, a_dtype: str = "bf16", out_dtype: str = "bf16",
                 *, in_elems: int = 0, out_elems: int = 0):
        self.M, self.N, self.K = M, N, K
        self.w_dtype, self.a_dtype, self.out_dtype = w_dtype, a_dtype, out_dtype
        self._in_elems = in_elems if in_elems else M * K
        self._out_elems = out_elems if out_elems else M * N
        super().__init__()

    @property
    def flops(self) -> float:
        return 2.0 * self.M * self.N * self.K

    @property
    def input_bytes(self) -> float:
        return self._in_elems * dtype_bytes(self.a_dtype)

    @property
    def weight_bytes(self) -> float:
        return (self.K * self.N * dtype_bytes(self.w_dtype)
                + gemm_scale_bytes(self.N, self.K, self.w_dtype))

    @property
    def output_bytes(self) -> float:
        return self._out_elems * dtype_bytes(self.out_dtype)


class TokenDispatch(Kernel):
    """MoE token dispatch: routing score + topk selection + token scatter.

    Softmax costs 5·M·N_experts; sigmoid costs 3·M·N_experts.
    bytes:
        input_bytes  = M·D·sizeof(a_dtype) + M·N_experts·4
        output_bytes = M·topk·D·sizeof(a_dtype)
    """

    def __init__(self, M: int, D: int, N_experts: int, topk: int,
                 a_dtype: str = "bf16", scoring_func: str = "softmax"):
        if scoring_func not in {"softmax", "sigmoid"}:
            raise ValueError(
                f"unsupported routing scoring function: {scoring_func}")
        self.M, self.D = M, D
        self.N_experts = N_experts
        self.topk = topk
        self.a_dtype = a_dtype
        self.scoring_func = scoring_func
        self.M_e = Fraction(M * topk, N_experts)
        super().__init__()

    @property
    def flops(self) -> float:
        score_flops = 5.0 if self.scoring_func == "softmax" else 3.0
        return score_flops * self.M * self.N_experts

    @property
    def input_bytes(self) -> float:
        return (self.M * self.D * dtype_bytes(self.a_dtype)
                + self.M * self.N_experts * dtype_bytes("fp32"))

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.M * self.topk * self.D * dtype_bytes(self.a_dtype)


class TokenCombine(Kernel):
    """MoE token combine: weighted sum of expert outputs.

    flops = 2·M·topk·D (multiply by routing weight + accumulate).
    bytes:
        input_bytes  = M·topk·D·sizeof(a_dtype)
        output_bytes = M·D·sizeof(a_dtype)
    """

    def __init__(self, M: int, D: int, N_experts: int, topk: int,
                 a_dtype: str = "bf16"):
        self.M, self.D = M, D
        self.N_experts = N_experts
        self.topk = topk
        self.a_dtype = a_dtype
        self.M_e = Fraction(M * topk, N_experts)
        super().__init__()

    @property
    def flops(self) -> float:
        return 2.0 * self.M * self.topk * self.D

    @property
    def input_bytes(self) -> float:
        return self.M * self.topk * self.D * dtype_bytes(self.a_dtype)

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.a_dtype)


class Sampling(Kernel):
    """Argmax / top-p sampling: select token IDs from logits.

    flops = M · V (comparisons for argmax / top-p selection).
    bytes:
        input_bytes  = M · V · sizeof(dtype)       (read logits)
        output_bytes = M · sizeof(out_dtype)        (write token IDs)
    """

    def __init__(self, M: int, V: int,
                 dtype: str = "bf16", out_dtype: str = "int32"):
        self.M, self.V = M, V
        self.dtype_, self.out_dtype = dtype, out_dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return float(self.M * self.V)

    @property
    def input_bytes(self) -> float:
        return self.M * self.V * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.M * dtype_bytes(self.out_dtype)
