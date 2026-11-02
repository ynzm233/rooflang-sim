from collections import Counter
import csv
from pathlib import Path

import pytest

from rooflang.programs.models.minimax_h3.audio_decode import audio_decode_nodes


def test_audio_decode_reconstructs_measured_leaf_counts_and_fp32():
    nodes = audio_decode_nodes({'target_audio_latents_per_channel': 207})
    assert Counter(n.operator.params.op_type for n in nodes) == {
        'Conv2DTranspose': 134, 'TransData': 1424, 'Conv2D': 256,
        'Sin': 127, 'Mul': 516, 'PadV3': 254, 'Slice': 254,
        'Add': 331, 'Pows': 127, 'MemSet': 254, 'LpNormV2': 135,
        'BroadcastTo': 254, 'Exp': 254, 'RealDiv': 142,
        'Reciprocal': 127, 'ClipByValueV2': 1,
    }
    assert len(nodes) == 4590
    assert len({n.node_name for n in nodes}) == len(nodes)
    assert all(n.operator.compute_dtype == 'fp32' for n in nodes)
    assert all(n.operator.params.output_dtypes == ('fp32',) for n in nodes
               if n.operator.params.op_type != 'MemSet')


@pytest.mark.parametrize('length', [1, 207, 414])
def test_audio_decode_geometry_scales_and_transpose_counts_input_work(length):
    nodes = audio_decode_nodes({'target_audio_latents_per_channel': length})
    by_name = {n.node_name: n.operator for n in nodes}
    assert by_name['AudioDecoder_Clamp'].params.output_shapes == ((2, 1, length * 800),)
    for level, factor in enumerate((5, 25, 50, 100, 200, 400, 800)):
        conv = by_name[f'AudioDecoder_L{level}_Upsample']
        attrs = conv.params.attributes
        assert attrs['logical_output_shape'][-1] == length * factor
        batch, cin, lin = attrs['logical_input_shape']
        _, cout, lout = attrs['logical_output_shape']
        assert conv.get_flops() == 2 * batch * lin * cin * cout * attrs['kernel'] + batch * cout * lout
    alias = by_name['AudioDecoder_OutputSnake_UpFilter']
    assert alias.params.attributes['logical_input_shape'] == [2, 8, length * 800 + 10]
    assert alias.params.attributes['logical_output_shape'] == [2, 8, length * 1600 + 30]
    assert alias.params.attributes['groups'] == 8
    residual = by_name['AudioDecoder_L0_K11_D5_Conv1']
    assert residual.params.attributes['dilation'] == 5
    assert residual.params.attributes['padding'] == 25
    assert residual.params.attributes['logical_output_shape'][-1] == length * 5


def test_audio_decode_rejects_invalid_length():
    for length in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match='positive integer'):
            audio_decode_nodes({'target_audio_latents_per_channel': length})


def test_audio_decode_matches_available_capture_tensor_boundaries():
    capture = Path(__file__).parents[1] / 'measured' / 'audio_vae_decode.csv'
    if not capture.exists():
        pytest.skip('Local measurement CSV is not distributed with the package')
    expected = Counter()
    with capture.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['op_type'] == 'MemSet':
                continue  # The capture reports N/A for this buffer shape.
            shapes = row['input_shapes'].strip('"').split(';')
            tensors = [(tuple(map(int, shape.split(','))) if shape else (), dtype)
                       for shape, dtype in zip(shapes, row['input_dtypes'].split(';'))]
            outputs = tuple(tuple(map(int, shape.split(',')))
                            for shape in row['output_shapes'].strip('"').split(';'))
            expected[(row['op_type'], tuple(sorted(tensors)), outputs)] += int(row['calls'])
    actual = Counter()
    dtype_names = {'fp32': 'FLOAT', 'int32': 'INT32', 'int64': 'INT64'}
    for node in audio_decode_nodes({'target_audio_latents_per_channel': 207}):
        p = node.operator.params
        if p.op_type == 'MemSet':
            continue
        tensors = [(shape, dtype_names[dtype]) for shape, dtype in zip(
            p.input_shapes + p.weight_shapes, p.input_dtypes + p.weight_dtypes)]
        actual[(p.op_type, tuple(sorted(tensors)), p.output_shapes)] += 1
    assert actual == expected
