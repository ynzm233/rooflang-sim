"""Analytical Ascend FIA subset, not an executable NPU kernel.

Scale1 + softmax14 and single-pass boundary traffic are analytical assumptions.
Wrapper copies, tiling, workspace and hardware overlap are not inferred.
"""
from dataclasses import dataclass
import math

from .core import ComputeComponent, Operator, OperatorParams, OpType, Registry


@dataclass
class NpuFusedInferAttentionScoreParams(OperatorParams):
    # TND: lengths are total packed storage rows, batch_size must be 1.
    # BNSD: lengths are per-batch storage rows.
    batch_size: int = 1
    num_heads: int = 1
    num_key_value_heads: int = 0
    q_len: int = 1
    kv_len: int = 1
    head_dim: int = 128
    input_layout: str = 'TND'
    actual_seq_lengths: list[int] | None = None
    actual_seq_lengths_kv: list[int] | None = None
    scale: float = 1.0
    sparse_mode: int = 0
    pre_tokens: int = 2147483647
    next_tokens: int = 2147483647
    inner_precise: int = 0
    softmax_lse_flag: bool = False
    # Describes SGLang's boolean upper-triangular atten_mask, not sparse=2/3.
    causal: bool = False

    def validate(self):
        for name in ('batch_size', 'num_heads', 'q_len', 'kv_len', 'head_dim'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.num_key_value_heads) is not int or self.num_key_value_heads < 0:
            raise ValueError('num_key_value_heads must be a nonnegative integer')
        if self.num_heads % (self.num_key_value_heads or self.num_heads):
            raise ValueError('num_heads must be divisible by num_key_value_heads')
        if self.input_layout not in ('TND', 'BNSD'):
            raise ValueError('only TND and BNSD layouts are modeled')
        for name, expected in (('sparse_mode', 0), ('inner_precise', 0),
                               ('pre_tokens', 2147483647), ('next_tokens', 2147483647)):
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ValueError(f'only {name}={expected} is modeled')
        if type(self.scale) not in (int, float) or not math.isfinite(self.scale):
            raise ValueError('scale must be finite')
        if type(self.causal) is not bool or type(self.softmax_lse_flag) is not bool:
            raise ValueError('causal and softmax_lse_flag must be booleans')
        if self.causal and (self.input_layout != 'BNSD' or self.q_len != self.kv_len):
            raise ValueError('causal models an explicit square BNSD self-attention mask')
        if self.input_layout == 'TND' and self.batch_size != 1:
            raise ValueError('TND has no storage batch dimension; use batch_size=1')


@Registry.register_operator('npu_fused_infer_attention_score', op_type=OpType.COMPUTE)
class NpuFusedInferAttentionScoreOperator(Operator):
    PARAM_CLASS = NpuFusedInferAttentionScoreParams
    COMPUTE_UNIT = 'mixed'
    source_url = 'https://gitee.com/ascend/cann-ops-adv/blob/master/docs/FusedInferAttentionScore.md'
    vector_compute_dtype = 'fp32'
    assumption = ('Ascend unquantized high-precision FIA; scale1 + softmax14; '
                  'cube uses input fp16/bf16, vector uses fp32; single-pass HBM '
                  'Q/K/V assigned to cube, O/mask/LSE to vector; serial cube+vector '
                  'compute roofline; score/probability workspace traffic, L2 effects, '
                  'wrapper copies, tile rereads and kernel overlap uncalibrated')

    def parse_args(self, **kwargs):
        p = self.params
        p.validate()
        if self.compute_dtype not in ('fp16', 'bf16'):
            raise ValueError('only unquantized fp16/bf16 FIA is modeled')
        self.kv_heads = p.num_key_value_heads or p.num_heads
        self.q_lengths = self._lengths(p.actual_seq_lengths, p.q_len, 'actual_seq_lengths')
        self.kv_lengths = self._lengths(p.actual_seq_lengths_kv, p.kv_len, 'actual_seq_lengths_kv')
        if len(self.q_lengths) != len(self.kv_lengths):
            raise ValueError('query and KV must describe the same number of documents')
        if p.causal and self.q_lengths != self.kv_lengths:
            raise ValueError('causal self-attention requires equal query and KV lengths')
        self.valid_pairs = sum(q * (q + 1) // 2 if p.causal else q * k
                               for q, k in zip(self.q_lengths, self.kv_lengths))

    def _lengths(self, values, storage_length, name):
        p = self.params
        if values is None:
            if p.input_layout == 'TND':
                raise ValueError(f'TND requires cumulative {name}')
            return [storage_length] * p.batch_size
        if (not isinstance(values, list) or not values
                or any(type(v) is not int or v <= 0 for v in values)):
            raise ValueError(f'{name} must be a nonempty list of positive integers')
        if p.input_layout == 'TND':
            # Empty documents have already been removed by the SGLang wrapper.
            if values[-1] != storage_length or any(a >= b for a, b in zip(values, values[1:])):
                raise ValueError(f'{name} must increase and end at the TND storage length')
            return [b - a for a, b in zip([0] + values[:-1], values)]
        if len(values) != p.batch_size or any(v > storage_length for v in values):
            raise ValueError(f'{name} must have one length per BNSD batch within storage')
        return list(values)

    @property
    def atten_mask_shape(self):
        p = self.params
        return [1, p.q_len, p.kv_len] if p.causal else []

    def get_cube_flops(self):
        return 4 * self.params.num_heads * self.params.head_dim * self.valid_pairs

    def get_vector_flops(self):
        # Official formula: Softmax(scale * QK^T) * V. Count scale even when
        # scale == 1; this model does not assume constant-folding by the kernel.
        # Softmax: max/sum reductions(2), subtract(1), exp(10), divide(1).
        return 15 * self.params.num_heads * self.valid_pairs

    def get_flops(self):
        return self.get_cube_flops() + self.get_vector_flops()

    def get_compute_component(self):
        return ComputeComponent(unit_type='mixed', cube_flops=self.get_cube_flops(),
                                vector_flops=self.get_vector_flops(), dtype=self.compute_dtype,
                                cube_dtype=self.compute_dtype,
                                vector_dtype=self.vector_compute_dtype)

    def get_memory_bytes_by_unit(self):
        """Boundary HBM traffic by consuming/producing unit, without tile rereads.

        FP32 vector arithmetic does not change Q/K/V/O storage. Score/probability
        workspace transfers and their L2 hit rate are outside this boundary model.
        Unit attribution is a roofline assumption, not a measured CANN trace.
        """
        p = self.params
        size = self.quant_config.get_compute_size()
        output_rows = p.batch_size * p.q_len
        qkv_bytes = (sum(self.q_lengths) * p.num_heads
                     + 2 * sum(self.kv_lengths) * self.kv_heads) * p.head_dim * size
        output_bytes = output_rows * p.num_heads * p.head_dim * size
        mask_bytes = math.prod(self.atten_mask_shape) if p.causal else 0
        lse_bytes = output_rows * p.num_heads * 4 if p.softmax_lse_flag else 4
        return {'cube': qkv_bytes, 'vector': output_bytes + mask_bytes + lse_bytes}

    def get_memory_details(self):
        return self.memory_details(activation=sum(self.get_memory_bytes_by_unit().values()))
