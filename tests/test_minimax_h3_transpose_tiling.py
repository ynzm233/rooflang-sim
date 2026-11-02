"""Small independent scatter/panel oracles, not measured-latency fixtures."""
import math
import random

import pytest

from rooflang.programs.models.minimax_h3.transpose_tiling import (
    KT, MT, NT, TransposeGeometry, transpose_cost,
)


def enumerated_cost(g, cores):
    """Enumerate outputs and source addresses; deliberately slow test oracle."""
    out = (g.length - 1) * g.stride - 2 * g.padding + g.dilation * (g.kernel - 1) + 1
    # Scatter from input, independent of the model's output-gather equations.
    pairs = sum(0 <= i * g.stride - g.padding + r * g.dilation < out
                for i in range(g.length) for r in range(g.kernel))
    ci, co = g.cin // g.groups, g.cout // g.groups
    result = dict(valid_pairs=pairs, input_read_bytes=0, filter_read_bytes=0,
                  bias_read_bytes=0, executed_cube_flops=0, executed_vector_flops=0,
                  critical_core_cube_flops=0, critical_core_vector_flops=0)
    for phase in range(min(g.stride, out)):
        ys = list(range(phase, out, g.stride))
        taps = [r for r in range(g.kernel)
                if (phase + g.padding - r * g.dilation) % g.stride == 0]
        reduction = [(r, c) for r in taps for c in range(ci)]
        cubes, vectors = [], []
        for _b in range(g.batch):
            for _group in range(g.groups):
                for start in range(0, len(ys), MT):
                    outputs = ys[start:start + MT]
                    for nc in range(0, co, NT):
                        columns = min(NT, co - nc)
                        cubes.append(2 * MT * NT * math.ceil(len(reduction) / KT) * KT)
                        vectors.append(len(outputs) * columns if g.bias else 0)
                        result['bias_read_bytes'] += columns * 4 if g.bias else 0
                        for k in range(0, len(reduction), KT):
                            panel = reduction[k:k + KT]
                            sources = set()
                            for y in outputs:
                                for r, c in panel:
                                    i = (y + g.padding - r * g.dilation) // g.stride
                                    if 0 <= i < g.length:
                                        sources.add((i, c))
                            result['input_read_bytes'] += len(sources) * 4
                            result['filter_read_bytes'] += len(panel) * columns * 4
        result['executed_cube_flops'] += sum(cubes)
        result['executed_vector_flops'] += sum(vectors)
        result['critical_core_cube_flops'] += sum(max(cubes[w:w + cores])
                                                 for w in range(0, len(cubes), cores))
        result['critical_core_vector_flops'] += sum(max(vectors[w:w + cores])
                                                   for w in range(0, len(vectors), cores))
    result['output_write_bytes'] = g.batch * math.ceil(g.cout / 8) * 8 * out * 4
    result['valid_flops'] = 2 * g.batch * g.cin * co * pairs + result['executed_vector_flops']
    return result


@pytest.mark.parametrize('g', [
    TransposeGeometry(1, 1, 1, 3, 3, stride=2, padding=1),
    TransposeGeometry(2, 4, 6, 17, 5, stride=3, padding=2, dilation=2, groups=2),
    TransposeGeometry(2, 8, 8, 7, 12, stride=2, groups=8, bias=False),
    TransposeGeometry(2, 9, 17, 19, 3, stride=1, padding=2),
    TransposeGeometry(1, 2, 3, 9, 1, stride=4),  # phases with no source contribution
    TransposeGeometry(1, 8, 16, 16, 1, bias=False),
])
@pytest.mark.parametrize('cores', [1, 3, 20])
def test_phase_tiles_match_independent_scatter_and_panel_oracle(g, cores):
    cost = transpose_cost(g, cores)
    for key, value in enumerated_cost(g, cores).items():
        assert getattr(cost, key) == value, key
    assert cost.reference_flops == (
        2 * g.batch * g.length * g.cin * (g.cout // g.groups) * g.kernel
        + (g.batch * g.cout * g.output_length if g.bias else 0))
    assert cost.activation_bytes == (cost.input_read_bytes + cost.filter_read_bytes
                                     + cost.output_write_bytes + 16)
    assert cost.workspace_bytes == 0


def test_random_padding_dilation_groups_and_vector_wave_tails():
    rng = random.Random(731)
    for _ in range(30):
        groups = rng.choice([1, 2, 3])
        g = TransposeGeometry(rng.randint(1, 3), groups * rng.randint(1, 9),
                              groups * rng.randint(1, 19), rng.randint(8, 33),
                              rng.randint(1, 6), stride=rng.randint(1, 4),
                              padding=rng.randint(0, 3), dilation=rng.randint(1, 3),
                              groups=groups, bias=True)
        cores = rng.randint(1, 23)
        cost = transpose_cost(g, cores)
        assert all(getattr(cost, k) == v for k, v in enumerated_cost(g, cores).items())


@pytest.mark.parametrize('batch,waves', [(19, 1), (20, 1), (21, 2)])
def test_wave_time_does_not_inflate_executed_flops(batch, waves):
    cost = transpose_cost(TransposeGeometry(batch, 8, 16, 16, 1, bias=False), 20)
    assert cost.executed_cube_flops == batch * 4096
    assert cost.critical_core_cube_flops == waves * 4096
    # 20 TFLOP/s aggregate means 1 TFLOP/s per core, with no added efficiency.
    assert cost.compute_time_us({'fp32:cube': 20, 'fp32:vector': 2}) == pytest.approx(waves * 4096 / 1e6)


def test_long_length_cost_is_compact_and_has_no_measured_time_dependency(monkeypatch):
    monkeypatch.setattr('builtins.open', lambda *a, **k: pytest.fail('prediction read a file'))
    g = TransposeGeometry(2, 8, 8, 10**9, 12, stride=2, groups=8, bias=False)
    cost = transpose_cost(g, 20)
    assert len(cost.phases) == 2
    assert cost.executed_cube_flops > cost.valid_flops
    assert cost.to_dict()['tile'] == [16, 16, 8]
    assert cost.to_dict()['calibration'] == 'none'


def test_large_bias_schedule_keeps_vector_tails_without_enumerating_tasks():
    # Alternating 256/16-element tiles (N=17); each full 20-core wave has max 256.
    cost = transpose_cost(TransposeGeometry(2, 8, 17, 10**9, 1), 20)
    tasks = 2 * (10**9 // 16) * 2
    assert cost.executed_vector_flops == 2 * 17 * 10**9
    assert cost.critical_core_vector_flops == math.ceil(tasks / 20) * 256
    # Large batch repeats the same single full tile; final wave is still explicit.
    cost = transpose_cost(TransposeGeometry(10**9 + 1, 8, 16, 16, 1), 20)
    assert cost.critical_core_vector_flops == math.ceil((10**9 + 1) / 20) * 256


@pytest.mark.parametrize('kwargs', [
    {'groups': 3}, {'stride': 0}, {'length': True}, {'padding': -1},
    {'padding': 100}, {'dilation': 0}, {'cin': 0}, {'bias': 1},
])
def test_invalid_geometry_is_rejected(kwargs):
    fields = dict(batch=2, cin=8, cout=8, length=5, kernel=3)
    fields.update(kwargs)
    with pytest.raises(ValueError):
        transpose_cost(TransposeGeometry(**fields), 20)


@pytest.mark.parametrize('cores', [0, -1, True, 1.5])
def test_invalid_core_count_is_rejected(cores):
    with pytest.raises(ValueError):
        transpose_cost(TransposeGeometry(1, 1, 1, 4, 3), cores)


def test_audio_bridge_activates_only_transpose_and_preserves_capacity(tmp_path, monkeypatch):
    import csv
    import json
    from pathlib import Path
    from rooflang.programs.models.minimax_h3.audio_transpose import AudioTransposeOperator
    from rooflang.programs.models.minimax_h3.bridge import build_component
    from rooflang.programs.presets.ascend910b import Ascend910B3
    from rooflang.runtime.simulator import Simulator
    from rooflang.runtime.metrics_export import export_metrics

    hw = Ascend910B3(1)
    with monkeypatch.context() as guard:
        def no_files(*args, **kwargs):
            pytest.fail('prediction must not read measurement or other files')
        guard.setattr('builtins.open', no_files)
        guard.setattr(Path, 'open', no_files)
        graph, placement = build_component('audio_vae_decoder', {},
            {'target_audio_latents_per_channel': 207}, 207, hw)
        result = Simulator(graph, placement, hw).run()
    assert len(graph.kernels) == 4590
    tiled = [k for k in graph.kernels if isinstance(k.operator, AudioTransposeOperator)]
    assert len(tiled) == 134
    for k in tiled:
        op, cost = k.operator, k.operator.tile_cost
        assert k.flops == cost.reference_flops
        assert sum(k.flops_by_dtype.values()) == cost.executed_cube_flops + cost.executed_vector_flops
        assert k.csv_metadata['activation_peak_bytes_proxy'] == op.activation_bytes
        assert sum(t.storage_bytes for t in k.weights.values()) == op.weight_saved_bytes
        assert k.loaded_weight_bytes == cost.bias_read_bytes
        assert k.outputs['traffic'].storage_bytes == 0
        assert k.csv_metadata['transpose_schedule']['calibration'] == 'none'
        device = placement.get_kernel_device(k).device
        assert Simulator.compute_time_us(k, device) == cost.compute_time_us(device.tflops)
    for k in graph.kernels:
        if k not in tiled:
            assert k.scheduled_compute_time_us(placement.get_kernel_device(k)) is None
    export_metrics({'audio_vae_decode': result}, tmp_path, model_name='minimax_h3')
    with (tmp_path / 'audio_vae_decode.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4590
    for row, entry in zip(rows, result.trace):
        assert float(row['compute_ms']) == pytest.approx(entry.compute_time_us / 1000)
        if entry.kernel in tiled:
            assert json.loads(row['transpose_schedule'])['executed_cube_flops'] == entry.kernel.operator.tile_cost.executed_cube_flops
        else:
            expected = sum(work / (entry.device.tflops[dtype] * 1e6)
                           for dtype, work in entry.kernel.flops_by_dtype.items())
            assert entry.compute_time_us == pytest.approx(expected)


def test_other_hardware_explicitly_keeps_legacy_transpose_cost():
    from rooflang.programs.models.minimax_h3.bridge import build_component
    from rooflang.programs.presets.ascend950dt import Ascend950DTSuperChip

    graph, placement = build_component('audio_vae_decoder', {},
        {'target_audio_latents_per_channel': 1}, 1, Ascend950DTSuperChip(8))
    tiled = [k for k in graph.kernels if k.csv_metadata['operator'] == 'Conv2DTranspose']
    assert len(tiled) == 134
    for k in tiled:
        assert k.operator.tile_cost is None
        assert k.csv_metadata['transpose_tiling_status'] == 'not_applied_hardware_scope'
        assert k.scheduled_compute_time_us(placement.get_kernel_device(k)) is None
        assert k.outputs['traffic'].size_bytes == k.operator.activation_bytes


@pytest.mark.parametrize('invalid', [-1, float('nan'), float('inf'), True])
def test_runtime_rejects_invalid_scheduled_time(invalid):
    from rooflang.language.kernels.kernel import Kernel
    from rooflang.runtime.simulator import Simulator

    class BadSchedule(Kernel):
        def scheduled_compute_time_us(self, device):
            return invalid
    with pytest.raises(ValueError, match='scheduled compute'):
        Simulator.compute_time_us(BadSchedule(), None)
