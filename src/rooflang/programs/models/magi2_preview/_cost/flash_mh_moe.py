"""Fused head-wise expert gather/MLP/weighted-scatter boundary."""
from dataclasses import dataclass, field
import math

from .core import Operator, OperatorParams, OpType, Registry
from .core import QuantConfig
from .core import ComputeComponent


@dataclass
class FlashMHMoEParams(OperatorParams):
    tokens: int = field(default=1, metadata={"min": 1})
    num_heads: int = field(default=1, metadata={"min": 1})
    num_experts: int = field(default=1, metadata={"min": 1})
    top_k: int = field(default=1, metadata={"min": 1})
    head_dim: int = field(default=1, metadata={"min": 1})
    intermediate_size: int = field(default=1, metadata={"min": 1})
    active_experts_per_head: int | None = field(default=None, metadata={"min": 1})
    expert_token_counts: list[int] = field(default_factory=list, metadata={"desc": "Flattened [H,E] assignments; omitted means balanced scenario"})
    resident_heads: int | None = field(default=None, metadata={"min": 1})
    deterministic: bool = False
    compute_amplification: float = field(default=1.0, metadata={"min": 1})
    sigmoid_flops: float = field(default=10.0, metadata={"min": 0})
    additional_workspace_bytes: int = field(default=0, metadata={"min": 0, "desc": "Calibrated extra traffic, default fused boundary lower bound"})


@Registry.register_operator("flash_mh_moe", op_type=OpType.COMPUTE)
class FlashMHMoEOperator(Operator):
    """One complete expert operator; its internal operations are not graph nodes.

    Residency follows allocated experts, reads follow activated experts, and
    arithmetic follows assignments. No expanded [assignments, F] intermediate
    is charged for the fused boundary. Unknown tile rereads, routing imbalance,
    deterministic-scatter overhead and workspace are disclosed, not invented.
    """
    PARAM_CLASS = FlashMHMoEParams
    COMPUTE_UNIT = "mixed"

    def __init__(self, **kwargs):
        kwargs.setdefault("quant_config", QuantConfig(compute_dtype="bf16", weight_storage_dtype="bf16"))
        super().__init__(**kwargs)

    def parse_args(self, **kwargs):
        p = self.params
        p.validate()
        if p.top_k > p.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if not math.isfinite(p.compute_amplification) or p.compute_amplification < 1:
            raise ValueError("compute_amplification must be finite and >= 1")
        if not math.isfinite(p.sigmoid_flops) or p.sigmoid_flops < 0:
            raise ValueError("sigmoid_flops must be finite and nonnegative")
        self.assignments = p.tokens*p.num_heads*p.top_k
        if p.expert_token_counts:
            if len(p.expert_token_counts) != p.num_heads*p.num_experts or any(type(v) is not int or not 0 <= v <= p.tokens for v in p.expert_token_counts):
                raise ValueError("expert_token_counts must contain H*E integers between zero and tokens")
            if any(sum(p.expert_token_counts[h*p.num_experts:(h+1)*p.num_experts]) != p.tokens*p.top_k for h in range(p.num_heads)):
                raise ValueError("each head must route exactly tokens*top_k assignments")
            if p.active_experts_per_head is not None:
                raise ValueError("choose expert_token_counts or active_experts_per_head")
            self.active_experts = sum(v > 0 for v in p.expert_token_counts)
        else:
            active = p.active_experts_per_head or min(p.num_experts, p.tokens*p.top_k)
            if not p.top_k <= active <= min(p.num_experts,p.tokens*p.top_k):
                raise ValueError("active expert count must be feasible for tokens and top_k")
            self.active_experts = p.num_heads*active
        self.resident_heads = p.resident_heads or p.num_heads
        if self.resident_heads < p.num_heads:
            raise ValueError("resident_heads cannot be smaller than computing heads")
        self.assumption = ("fused expert boundary traffic lower bound; cube/vector share configured compute dtype; "
                           "deterministic scatter, tile rereads, imbalance and scratch uncalibrated; "
                           f"deterministic={p.deterministic}; compute_amplification={p.compute_amplification}")

    def get_cube_flops(self):
        return 6*self.assignments*self.params.head_dim*self.params.intermediate_size

    def get_vector_flops(self):
        from .swiglu7 import swiglu7_flops
        p = self.params
        return swiglu7_flops(self.assignments*p.intermediate_size,p.sigmoid_flops)+2*self.assignments*p.head_dim

    def get_flops(self):
        return self.get_cube_flops()+self.get_vector_flops()

    def get_compute_component(self):
        return ComputeComponent(unit_type="mixed",
                                cube_flops=self.get_cube_flops()*self.params.compute_amplification,
                                vector_flops=self.get_vector_flops(), dtype=self.compute_dtype)

    def get_memory_details(self):
        p = self.params
        weight_per_expert = 3*p.head_dim*p.intermediate_size*self.quant_config.get_weight_size()
        activation = (2*p.tokens*p.num_heads*p.head_dim*self.quant_config.get_compute_size()
                      +self.assignments*8+(p.num_heads*p.num_experts+1)*8+p.additional_workspace_bytes)
        return {"activation": activation, "weight_read": self.active_experts*weight_per_expert,
                "weight_saved": self.resident_heads*p.num_experts*weight_per_expert,
                "kv_cache_read": 0, "kv_cache_saved": 0, "kv_cache_load": 0}
