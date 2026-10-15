"""FA3 followed by source-visible sh-layout sink output/LSE correction."""
from dataclasses import dataclass, field

from .core import OpType, Registry
from .core import CompositeOperator, OperatorStage
from .flash_attention import FlashAttentionOperator, FlashAttentionParams


@dataclass
class FlashAttentionSinkParams(FlashAttentionParams):
    sink_tokens: int = field(default=1, metadata={"min": 1})
    sink_weight_heads: int | None = field(default=None, metadata={"min": 1, "desc": "Resident heads before any CP slice"})


@Registry.register_operator("flash_attention_sink", op_type=OpType.COMPUTE)
class FlashAttentionSinkOperator(CompositeOperator):
    """A public composite node, not a physically fused attention kernel.

    Correction uses explicit source-operation subcalls. torch.compile may fuse
    these; the default is an uncalibrated source-operation scenario. No CUDA
    implementation details are simulated.
    """
    PARAM_CLASS = FlashAttentionSinkParams
    COMPUTE_UNIT = "mixed"

    def parse_args(self, **kwargs):
        self.params.validate()
        if self.params.causal:
            raise ValueError("sink model currently supports noncausal packed attention")
        attention = {key: value for key, value in vars(self.params).items()
                     if key not in ("sink_tokens", "sink_weight_heads")}
        self.attention = FlashAttentionOperator(**attention, quant_config=self.quant_config)
        self.assumption = ("FA3 plus explicit FP32 sh-layout sink correction subcalls; "
                           "FA cube/vector share configured compute dtype; compiler fusion uncalibrated; "
                           f"softmax_compute_multiplier={self.params.softmax_compute_multiplier}")

    def get_stages(self):
        p, fa = self.params, self.attention
        # Engine node precision overrides are applied after construction.
        fa.quant_config = self.quant_config
        n = p.batch_size*p.q_len*p.num_heads
        output = n*p.head_dim*self.quant_config.get_compute_size()
        sink_elements = p.sink_tokens*p.num_heads
        saved_heads = p.sink_weight_heads or p.num_heads
        # Exp/log each use ten arithmetic-equivalent FLOPs, following the FA
        # convention. Reduction/safety scalar work is explicitly a proxy.
        return [
            OperatorStage("flash_attention", fa.get_cube_flops(), fa.get_vector_flops(),
                          fa.get_memory_bytes()+n*4, compute_dtype=self.compute_dtype,
                          vector_compute_multiplier=p.softmax_compute_multiplier),
            OperatorStage("sink_logsumexp", vector_flops=sink_elements*12+p.num_heads*11,
                          activation_bytes=p.num_heads*4, weight_read_bytes=sink_elements*4,
                          weight_saved_bytes=p.sink_tokens*saved_heads*4),
            OperatorStage("sink_repeat", activation_bytes=(p.num_heads+n)*4),
            OperatorStage("correct_lse", vector_flops=n*27, activation_bytes=n*12),
            OperatorStage("rescale_weight", vector_flops=n*15, activation_bytes=n*12),
            OperatorStage("scale_output", vector_flops=n*p.head_dim, activation_bytes=2*output+n*4),
            OperatorStage("copy_corrected_lse", activation_bytes=n*8),
            OperatorStage("lse_transpose_contiguous", activation_bytes=n*8),
        ]
