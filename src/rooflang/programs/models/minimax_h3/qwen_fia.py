"""Separate Qwen FIA costs; public-source tile candidates, not fitted NPU timing.

Semantic pairs, executed rectangles and padded Cube work are distinct. Boundary
HBM remains physical Q/K/V/O storage: padding is not an allocation or a reread.
"""
from dataclasses import dataclass

from rooflang.programs.models.magi2_preview._cost.npu_fused_infer_attention_score import (
    NpuFusedInferAttentionScoreOperator,
    NpuFusedInferAttentionScoreParams,
)


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


@dataclass
class _QwenFIAParams(NpuFusedInferAttentionScoreParams):
    q_tile: int = 128
    kv_tile: int = 128
    cube_alignment: int = 16

    def validate(self):
        super().validate()
        if type(self.cube_alignment) is not int or self.cube_alignment != 16:
            raise ValueError('Qwen FIA models Cube alignment=16 only')
        for name in ('q_tile', 'kv_tile'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0 or value % self.cube_alignment:
                raise ValueError(f'{name} must be a positive multiple of cube_alignment')


@dataclass
class QwenVisionFIAParams(_QwenFIAParams):
    head_dim: int = 72
    q_tile: int = 256

    def validate(self):
        super().validate()
        if self.input_layout != 'TND' or self.causal:
            raise ValueError('QwenVisionFIA requires noncausal TND attention')
        if self.num_key_value_heads not in (0, self.num_heads):
            raise ValueError('QwenVisionFIA requires equal Q and KV heads')


@dataclass
class QwenTextFIAParams(_QwenFIAParams):
    input_layout: str = 'BNSD'
    causal: bool = True
    kv_tile: int = 512
    execution_mode: str = 'full_matrix'

    def validate(self):
        super().validate()
        if self.input_layout != 'BNSD' or not self.causal:
            raise ValueError('QwenTextFIA requires causal BNSD self-attention')
        if self.execution_mode not in ('full_matrix', 'causal_blocks'):
            raise ValueError('execution_mode must be full_matrix or causal_blocks')


class _QwenFIA(NpuFusedInferAttentionScoreOperator):
    tiling_source_commit = 'e0f6933182bcc757ebcde538fbd635bbc7deadfb'
    assumption = (
        'Qwen-specific analytical FIA: semantic pairs differ from executed tiles; '
        'Cube rows/columns/D align to 16, vector uses active tile rectangles; '
        'scale1+softmax14 plus online row-rescale proxy; serial Cube+FP32 Vector; '
        'public-source tile candidates, not verified runtime tiling or latency fit; '
        'single-pass boundary HBM excludes tile rereads, workspace, L2 and wrapper copies'
    )

    def parse_args(self, **kwargs):
        super().parse_args(**kwargs)
        if self.q_lengths != self.kv_lengths:
            raise ValueError('Qwen FIA requires matching self-attention documents')
        p = self.params
        self.execution_mode = getattr(p, 'execution_mode', 'full_matrix')
        self.padded_head_dim = _align(p.head_dim, p.cube_alignment)
        self.executed_pairs = self.cube_pairs = 0
        self.tile_count = self.tail_tile_count = self.row_updates = 0
        for q_length, kv_length in zip(self.q_lengths, self.kv_lengths):
            for start in range(0, q_length, p.q_tile):
                rows = min(p.q_tile, q_length - start)
                # The diagonal block is executed in full; only blocks entirely
                # above the causal diagonal may be skipped in the opt-in mode.
                end = (min(kv_length, _align(start + rows, p.kv_tile))
                       if self.execution_mode == 'causal_blocks' else kv_length)
                tiles = (end + p.kv_tile - 1) // p.kv_tile
                self.executed_pairs += rows * end
                self.cube_pairs += _align(rows, p.cube_alignment) * _align(end, p.cube_alignment)
                self.tile_count += tiles
                full_tiles = end // p.kv_tile if rows == p.q_tile else 0
                self.tail_tile_count += tiles - full_tiles
                self.row_updates += rows * (tiles - 1)

    def get_cube_flops(self):
        return 4 * self.params.num_heads * self.padded_head_dim * self.cube_pairs

    def get_vector_flops(self):
        # Additional online merge: max/sub/exp/mul/add=14, plus 2D to rescale
        # and accumulate the output. This supplements the legacy 15/pair proxy.
        p = self.params
        return p.num_heads * (15 * self.executed_pairs + (2 * p.head_dim + 14) * self.row_updates)

    def cost_metadata(self):
        return {
            'attention_model': type(self).__name__,
            'attention_pairs': self.valid_pairs,
            'executed_attention_pairs': self.executed_pairs,
            'cube_attention_pairs': self.cube_pairs,
            'cube_padded_head_dim': self.padded_head_dim,
            'attention_tile_count_per_head': self.tile_count,
            'attention_tail_tiles_per_head': self.tail_tile_count,
            'attention_row_updates_per_head': self.row_updates,
            'attention_execution_mode': self.execution_mode,
            'gqa_group_size': self.params.num_heads // self.kv_heads,
            'qwen_tiling': {'q_tile': self.params.q_tile, 'kv_tile': self.params.kv_tile,
                            'cube_alignment': self.params.cube_alignment},
            'tiling_source_commit': self.tiling_source_commit,
            'tiling_status': 'public_source_candidate_runtime_unverified',
            'boundary_traffic_includes_tile_rereads': False,
        }


class QwenVisionFIA(_QwenFIA):
    """One TND call preserving every document boundary, including unequal tails."""
    PARAM_CLASS = QwenVisionFIAParams


class QwenTextFIA(_QwenFIA):
    """Explicit-mask GQA: full-matrix default, optional causal-block scenario."""
    PARAM_CLASS = QwenTextFIAParams
