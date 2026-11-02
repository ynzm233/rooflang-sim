from collections import Counter
import csv
from pathlib import Path

import pytest

from rooflang.programs.models.minimax_h3.video_decode import video_decode_nodes, video_decode_stitch_nodes


MM = {'target_height': 768, 'target_width': 1344, 'target_latent_frames': 37}


def test_video_decode_reconstructs_measured_layer_inventory_and_order():
    nodes = video_decode_nodes(1797, MM)
    expected = {'FlashAttentionScore': 1, 'MatMulV2': 1, 'MatMulV3': 3,
                'Slice': 10, 'Transpose': 3, 'Mul': 12, 'SwiGlu': 1,
                'Add': 8, 'ReduceMean': 4, 'Pows': 4, 'ConcatD': 4,
                'Cast': 6, 'AsStrided': 1, 'Neg': 2, 'Rsqrt': 4}
    for layer in range(1, 37):
        assert Counter(n.operator.params.op_type for n in nodes if n.layer_idx == layer) == expected
    assert len([n for n in nodes if n.layer_idx == 0]) == 28
    positions = {node.node_name: i for i, node in enumerate(nodes)}
    assert positions['VideoDecoder_XEmbedder'] < positions['VideoDecoder_L1_Norm1_Pows']
    assert positions['VideoDecoder_L36_Residual2'] < positions['VideoDecoder_FinalNorm']
    assert positions['VideoDecoder_FinalNorm'] < positions['VideoDecoder_ProjOut']
    assert len(positions) == len(nodes)


def test_video_decode_attention_dtype_and_partial_rope():
    nodes = video_decode_nodes(1797, MM)
    ops = {n.node_name: n.operator for n in nodes}
    attention = ops['VideoDecoder_L1_FlashAttention']
    assert attention.params.input_shapes == ((1, 32, 1797, 64),) * 3
    assert attention.params.input_dtypes == ('fp16',) * 3
    assert attention.compute_dtype == 'fp16'
    assert attention.vector_compute_dtype == 'fp32'
    assert attention.cube_flops == 4 * 32 * 1797**2 * 64
    assert ops['VideoDecoder_L1_Norm1_Pows'].params.input_dtypes == ('fp32',)
    assert ops['VideoDecoder_L1_QNorm_CastIn'].params.output_dtypes == ('fp32',)
    assert ops['VideoDecoder_L1_QRoPE_MulCos'].params.output_shapes == ((1, 1797, 32, 48),)
    assert ops['VideoDecoder_L1_LayerScale1'].params.input_dtypes == ('fp16',)
    assert ops['VideoDecoder_L1_LayerScale1'].params.weight_dtypes == ('fp32',)
    assert ops['VideoDecoder_L1_LayerScale1'].params.output_dtypes == ('fp32',)
    sliced = ops['VideoDecoder_L1_QSlice']
    assert sliced.get_memory_details()['activation'] == 2 * 1797 * 32 * 64 * 2


def test_video_decode_scaling_keeps_weights_and_uses_real_tile_grid():
    small_mm = {**MM, 'target_height': 128, 'target_width': 256}
    small = video_decode_nodes(901, small_mm)
    large = video_decode_nodes(1797, MM)
    saved = lambda ns: sum(n.operator.get_memory_details()['weight_saved'] for n in ns)
    assert saved(small) == saved(large)
    fa = lambda ns: next(n.operator for n in ns if n.operator.params.op_type == 'FlashAttentionScore')
    assert fa(small).cube_flops / fa(large).cube_flops == pytest.approx((901 / 1797)**2)
    conv = next(n.operator for n in small if n.operator.params.op_type == 'Conv3DV2')
    assert conv.params.input_shapes == ((1, 24, 7, 8, 16),)
    with pytest.raises(ValueError, match='sequence'):
        video_decode_nodes(902, small_mm)


def test_video_decode_stitch_recovers_all_remaining_kernels_and_view_traffic():
    nodes = video_decode_stitch_nodes(MM)
    assert Counter(n.operator.params.op_type for n in nodes) == {
        'ViewCopy': 204, 'Slice': 1154, 'Mul': 642, 'ConcatD': 321, 'Add': 321}
    assert all(n.operator.get_memory_details()['weight_saved'] == 0 for n in nodes)
    copies = [n.operator for n in nodes if n.operator.params.op_type == 'ViewCopy']
    final = copies[-1]
    assert final.params.output_shapes == ((3 * 124 * 768 * 1344,),)
    assert final.get_memory_details()['activation'] == 2 * 3 * 5 * 768 * 1344 * 4
    small = video_decode_stitch_nodes({**MM, 'target_height': 128, 'target_width': 128})
    assert Counter(n.operator.params.op_type for n in small) == {
        'ViewCopy': 15, 'Slice': 20, 'Mul': 12, 'ConcatD': 6, 'Add': 6}


def test_video_decode_all_measured_tensor_signatures_and_calls():
    source = Path(__file__).parents[1] / 'measured' / 'video_vae_decode.csv'
    if not source.exists():
        pytest.skip('Local measured trace is not distributed with the package')

    def signature(kind, inputs, outputs):
        return kind, tuple(sorted(inputs)), tuple(sorted(outputs))

    def measured_tensors(shapes, dtypes):
        dtype_map = {'FLOAT16': 'fp16', 'FLOAT': 'fp32'}
        return [(tuple(map(int, shape.split(','))), dtype_map[dtype])
                for shape, dtype in zip(shapes.strip('"').split(';'), dtypes.split(';'))
                if shape and dtype in dtype_map]

    measured = Counter()
    with source.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = signature(row['op_type'],
                            measured_tensors(row['input_shapes'], row['input_dtypes']),
                            measured_tensors(row['output_shapes'], row['output_dtypes']))
            measured[key] += int(row['calls'])
    modeled = Counter()
    for nodes, repeats in ((video_decode_nodes(1797, MM), 196), (video_decode_stitch_nodes(MM), 1)):
        for node in nodes:
            p = node.operator.params
            inputs = list(zip(p.input_shapes, p.input_dtypes)) + list(zip(p.weight_shapes, p.weight_dtypes))
            modeled[signature(p.op_type, inputs, zip(p.output_shapes, p.output_dtypes))] += repeats
    assert sum(modeled.values()) == 459714
    assert modeled == measured
