"""FlashAttention with dense, causal, explicit-range or 3-D window masks."""

from dataclasses import dataclass, field
import math
from typing import Any

from .core import Operator, OperatorParams, OpType, Registry, ComputeComponent
from .utils import causal_factor

# Softmax FLOPs: exp(10) + sub(1) + div(1) + reductions(2) = 14
_SOFTMAX_FLOPS_PER_ELEMENT = 14

# Two matmuls: QK^T and PV
_MATMUL_COUNT = 2


def window_attention_pairs(video_shape, text_tokens, block_shape, window_radius):
    """Exact pairs, including partial edge blocks and dense text interactions."""
    axis_pairs = []
    for dim, block, radius in zip(video_shape, block_shape, window_radius):
        sizes = [min(block, dim - start) for start in range(0, dim, block)]
        axis_pairs.append(sum(sizes[i] * sizes[j] for i in range(len(sizes))
                              for j in range(max(0, i-radius), min(len(sizes), i+radius+1))))
    video_tokens = math.prod(video_shape)
    return math.prod(axis_pairs) + 2 * video_tokens * text_tokens + text_tokens**2


def range_attention_pairs(q_ranges, kv_ranges, q_len, kv_len):
    """Count the union of range rectangles; overlapping masks never double count."""
    if len(q_ranges) != len(kv_ranges) or not q_ranges:
        raise ValueError("q_ranges and kv_ranges must be nonempty equal-length lists")
    rectangles = []
    for q, k in zip(q_ranges, kv_ranges):
        if len(q) != 2 or len(k) != 2 or not (0 <= q[0] < q[1] <= q_len and 0 <= k[0] < k[1] <= kv_len):
            raise ValueError("attention ranges must be nonempty half-open intervals within sequence lengths")
        if any(type(v) is not int for v in (*q, *k)):
            raise ValueError("attention range boundaries must be integers")
        rectangles.append((*q, *k))
    edges = sorted({edge for q0, q1, _, _ in rectangles for edge in (q0, q1)})
    pairs = 0
    for q0, q1 in zip(edges, edges[1:]):
        intervals = sorted((k0, k1) for a, b, k0, k1 in rectangles if a <= q0 and b >= q1)
        covered = 0
        end = 0
        for k0, k1 in intervals:
            covered += max(0, k1-max(k0, end))
            end = max(end, k1)
        pairs += (q1-q0)*covered
    return pairs


@dataclass
class FlashAttentionParams(OperatorParams):
    """FlashAttention parameters."""

    batch_size: int = field(default=1, metadata={"min": 1, "desc": "Batch size (B)"})
    num_heads: int = field(default=1, metadata={"min": 1, "desc": "Number of attention heads (H_Q)"})
    kv_heads: int = field(default=1, metadata={"min": 1, "desc": "Number of kv heads (H_KV)"})
    q_len: int = field(default=1, metadata={"min": 1, "desc": "Q Sequence length (S_Q)"})
    kv_len: int = field(default=1, metadata={"min": 1, "desc": "KV Sequence length (S_KV)"})
    head_dim: int = field(default=64, metadata={"min": 1, "desc": "Head dimension (D)"})
    causal: bool = field(default=False, metadata={"desc": "Whether to use causal masking"})
    attention_pairs: int | None = field(default=None, metadata={"min": 0, "desc": "Valid QK pairs per batch, excluding heads"})
    video_shape: list[int] = field(default_factory=list, metadata={"desc": "Video token grid [T,H,W]"})
    text_tokens: int = field(default=0, metadata={"min": 0})
    block_shape: list[int] = field(default_factory=lambda: [8, 4, 4])
    window_radius: list[int] = field(default_factory=lambda: [2, 2, 2])
    q_ranges: list[list[int]] = field(default_factory=list)
    kv_ranges: list[list[int]] = field(default_factory=list)
    softmax_compute_multiplier: float = field(default=1.0, metadata={"min": 0})


@Registry.register_operator("flash_attention", op_type=OpType.COMPUTE)
class FlashAttentionOperator(Operator):
    """One fused attention boundary with the existing 14-FLOP softmax rule.

    Optional masks change mathematical pairs, while softmax_compute_multiplier
    changes only the timing proxy. Q/K/V/O traffic remains the tensor boundary
    lower bound; mask padding, kernel rereads and workspace are not inferred.
    """

    COMPUTE_UNIT = "mixed"
    PARAM_CLASS = FlashAttentionParams

    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.batch_size = params.batch_size if params is not None else kwargs.get("batch_size", 0)
        self.num_heads = params.num_heads if params is not None else kwargs.get("num_heads", 0)
        self.kv_heads = params.kv_heads if params is not None else kwargs.get("kv_heads", 0)
        self.q_len = params.q_len if params is not None else kwargs.get("q_len", 0)
        self.kv_len = params.kv_len if params is not None else kwargs.get("kv_len", 0)
        self.head_dim = params.head_dim if params is not None else kwargs.get("head_dim", 0)
        self.causal = params.causal if params is not None else kwargs.get("causal", False)
        p = self.params
        if not math.isfinite(p.softmax_compute_multiplier) or p.softmax_compute_multiplier <= 0:
            raise ValueError("softmax_compute_multiplier must be finite and positive")
        masks = sum((p.attention_pairs is not None, bool(p.video_shape), bool(p.q_ranges or p.kv_ranges)))
        if masks > 1:
            raise ValueError("choose attention_pairs, video_shape, or explicit ranges")
        if masks and p.causal:
            raise ValueError("explicit mask already defines its pairs; causal must be false")
        if masks and self.num_heads % self.kv_heads:
            raise ValueError("num_heads must be divisible by kv_heads")
        if p.video_shape:
            if any(len(v) != 3 for v in (p.video_shape, p.block_shape, p.window_radius)):
                raise ValueError("video_shape, block_shape and window_radius need three axes")
            if any(type(v) is not int or v <= 0 for v in p.video_shape+p.block_shape) or any(type(v) is not int or v < 0 for v in p.window_radius):
                raise ValueError("video/block axes must be positive integers; radii nonnegative")
            if math.prod(p.video_shape)+p.text_tokens != self.q_len or self.q_len != self.kv_len:
                raise ValueError("window sequence lengths must equal video plus text tokens")
            self.valid_pairs = window_attention_pairs(p.video_shape, p.text_tokens, p.block_shape, p.window_radius)
        elif p.q_ranges or p.kv_ranges:
            self.valid_pairs = range_attention_pairs(p.q_ranges, p.kv_ranges, self.q_len, self.kv_len)
        else:
            self.valid_pairs = p.attention_pairs if p.attention_pairs is not None else self.q_len*self.kv_len
        if type(self.valid_pairs) is not int or not 0 <= self.valid_pairs <= self.q_len*self.kv_len:
            raise ValueError("attention_pairs must be an integer within the dense matrix")
        if masks or p.softmax_compute_multiplier != 1:
            self.assumption = ("valid mask pairs; FA softmax14; QKVO traffic lower bound; "
                               "shared compute dtype for cube/vector throughput; "
                               f"softmax_compute_multiplier={p.softmax_compute_multiplier}")

    @property
    def _causal_factor(self) -> float:
        return causal_factor(self.causal, self.q_len, self.kv_len)

    @property
    def _base_elements(self) -> int:
        return self.batch_size * self.num_heads * self.q_len * self.head_dim

    @property
    def _kv_elements(self) -> int:
        return self.batch_size * self.kv_heads * self.kv_len * self.head_dim

    @property
    def _attn_elements(self) -> int:
        return self.batch_size * self.num_heads * self.valid_pairs

    def get_flops(self) -> float:
        matmul_flops = _MATMUL_COUNT * 2 * self._attn_elements * self.head_dim
        softmax_flops = _SOFTMAX_FLOPS_PER_ELEMENT * self._attn_elements
        return (matmul_flops + softmax_flops) * self._causal_factor

    def get_cube_flops(self) -> float:
        return _MATMUL_COUNT * 2 * self._attn_elements * self.head_dim * self._causal_factor

    def get_vector_flops(self) -> float:
        return _SOFTMAX_FLOPS_PER_ELEMENT * self._attn_elements * self._causal_factor

    def get_memory_details(self) -> dict[str, int]:
        # Q read + O write: always from activation
        activation_bytes = self._base_elements * self.quant_config.get_compute_size() * 2
        if self.causal:
            # LLM decode: K and V read from kv_cache (persistent across decode steps)
            kv_cache_bytes = self._kv_elements * self.quant_config.get_kv_size() * 2
            return self.memory_details(activation=activation_bytes, kv_cache=kv_cache_bytes)
        else:
            # Encoder / cross-attn: K and V are transient activations, no kv_cache
            activation_bytes += self._kv_elements * self.quant_config.get_compute_size() * 2
            return self.memory_details(activation=activation_bytes, kv_cache=0)

    def get_compute_component(self) -> ComputeComponent:
        return ComputeComponent(
            unit_type="mixed",
            cube_flops=self.get_cube_flops(),
            vector_flops=self.get_vector_flops() * self.params.softmax_compute_multiplier,
            dtype=self.compute_dtype,
        )
