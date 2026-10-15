"""Head-wise expert routing and CSR assignment construction.

Source boundary: MAGI route.py compute_topk_probs_and_indices followed by
flash_mh_moe_global_sort. Router-logits GEMM and EP communication are excluded.
"""
from dataclasses import dataclass, field
import math

from .core import OperatorParams, OpType, Registry
from .core import QuantConfig
from .core import CompositeOperator, OperatorStage
from .sigmoid import SigmoidOperator
from .softmax import SoftmaxOperator
from .topk import TopkOperator


@dataclass
class MoERoutingParams(OperatorParams):
    tokens: int = field(default=1, metadata={"min": 1})
    num_heads: int = field(default=1, metadata={"min": 1})
    num_experts: int = field(default=1, metadata={"min": 1})
    top_k: int = field(default=1, metadata={"min": 1})
    score_func: str = field(default="sigmoid", metadata={"choices": ["sigmoid", "softmax"]})
    route_norm: bool = True
    has_bias: bool = True
    sort_workspace_bytes: int = field(default=0, metadata={"min": 0, "desc": "Additional sort traffic, uncalibrated by default"})


@Registry.register_operator("moe_routing", op_type=OpType.COMPUTE)
class MoERoutingOperator(CompositeOperator):
    """One composite node, preserving source operation boundaries internally.

    Supports the released ungrouped route with probabilities gathered from the
    unbiased scores. Stable-sort comparisons use N*ceil(log2 N); the actual
    sorting algorithm and scratch accesses need device calibration.
    """
    PARAM_CLASS = MoERoutingParams
    COMPUTE_UNIT = "vector"

    def __init__(self, **kwargs):
        kwargs.setdefault("quant_config", QuantConfig(compute_dtype="fp32", weight_storage_dtype="fp32"))
        super().__init__(**kwargs)

    def parse_args(self, **kwargs):
        p = self.params
        p.validate()
        if p.top_k > p.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.assumption = "source-operation roofline sum; ungrouped routing; stable-sort comparison proxy and minimum traffic; scratch/fusion uncalibrated"

    def get_stages(self):
        p = self.params
        rows = p.tokens*p.num_heads
        logits, assignments = rows*p.num_experts, rows*p.top_k
        experts = p.num_heads*p.num_experts
        size = self.quant_config.get_compute_size()
        dtype = self.compute_dtype
        score_cls = SigmoidOperator if p.score_func == "sigmoid" else SoftmaxOperator
        score = score_cls(batch_seq=rows, dim=p.num_experts, quant_config=self.quant_config)
        topk = TopkOperator(batch_seq=rows, dim=p.num_experts, k=p.top_k, quant_config=self.quant_config)
        stages = [OperatorStage("scores_"+p.score_func, vector_flops=score.get_flops(),
                                activation_bytes=2*logits*size, compute_dtype=dtype)]
        if p.has_bias:
            stages.append(OperatorStage("selection_bias", vector_flops=logits,
                                        activation_bytes=2*logits*size,
                                        weight_read_bytes=experts*size, weight_saved_bytes=experts*size,
                                        compute_dtype=dtype))
        stages.extend([
            # torch.topk materializes values AND int64 indices, although only
            # its indices are subsequently used to gather unbiased scores.
            OperatorStage("topk", vector_flops=topk.get_flops(),
                          activation_bytes=logits*size+assignments*(size+8), compute_dtype=dtype),
            OperatorStage("gather_unbiased_probs", activation_bytes=assignments*(8+2*size), compute_dtype=dtype),
        ])
        if p.route_norm:
            stages.extend([
                OperatorStage("l1_norm", vector_flops=2*assignments,
                              activation_bytes=(assignments+rows)*size, compute_dtype=dtype),
                OperatorStage("norm_clamp_min", vector_flops=rows,
                              activation_bytes=2*rows*size, compute_dtype=dtype),
                OperatorStage("normalize_probs", vector_flops=assignments,
                              activation_bytes=(2*assignments+rows)*size, compute_dtype=dtype),
            ])
        stages.extend([
            OperatorStage("head_arange", activation_bytes=p.num_heads*8),
            OperatorStage("head_offsets", vector_flops=p.num_heads, activation_bytes=2*p.num_heads*8),
            OperatorStage("global_expert_ids", vector_flops=assignments,
                          activation_bytes=(2*assignments+p.num_heads)*8),
            OperatorStage("token_arange", activation_bytes=p.tokens*8),
            OperatorStage("expand_token_ids", activation_bytes=(p.tokens+assignments)*8),
            OperatorStage("stable_argsort", vector_flops=assignments*math.ceil(math.log2(max(2,assignments))),
                          activation_bytes=assignments*16+p.sort_workspace_bytes),
            OperatorStage("gather_token_ids", activation_bytes=assignments*24),
            OperatorStage("cast_gather_ids_int32", activation_bytes=assignments*12),
            OperatorStage("gather_sorted_probs", activation_bytes=assignments*(8+2*size), compute_dtype=dtype),
        ])
        if dtype != "fp32":
            stages.append(OperatorStage("cast_sorted_probs_fp32", activation_bytes=assignments*(size+4)))
        stages.extend([
            OperatorStage("zero_expert_counts", activation_bytes=experts*8),
            OperatorStage("gather_sorted_expert_ids", activation_bytes=assignments*24),
            OperatorStage("ones_for_counts", activation_bytes=assignments*8),
            OperatorStage("scatter_add_counts", vector_flops=assignments,
                          activation_bytes=assignments*32),
            OperatorStage("zero_expert_offsets", activation_bytes=(experts+1)*8),
            OperatorStage("cumsum_counts", vector_flops=max(0,experts-1), activation_bytes=experts*16),
            OperatorStage("copy_expert_offsets", activation_bytes=experts*16),
        ])
        return stages
