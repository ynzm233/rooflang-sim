"""Hand-calculated timeline and physical residency checks for layer offload."""
import json
from pathlib import Path

import pytest

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.programs.models.minimax_h3.offload import WeightLoad, apply_layerwise_offload, graph_stats
from rooflang.programs.models.minimax_h3.workflow import create_workflow
from rooflang.programs.presets.ascend910b import Ascend910B3
from rooflang.runtime.simulator import OOMError, Simulator


class ToyLayer(Kernel):
    def __init__(self, layer, compute_us, weight_bytes=32_000):
        super().__init__(weights={'w': Tensor('bf16', (weight_bytes // 2,), weight_id=f'w{layer}')}
                         if weight_bytes else {})
        self.compute_us = compute_us
        self.csv_metadata = {'layer_idx': layer, 'operator': 'ToyLayer'}

    @property
    def flops_by_dtype(self):
        return {'bf16:cube': self.compute_us * 294_912_000}


def toy(prefetch=1, *, weight_bytes=32_000, compute_us=2, split=False):
    hardware = Ascend910B3()
    graph = ComputeGraph()
    placement = Placement(hardware, graph)
    kernels = []
    for layer in range(3):
        pieces = [ToyLayer(layer, compute_us, weight_bytes)]
        if split:
            pieces = [ToyLayer(layer, 0, weight_bytes), ToyLayer(layer, compute_us, 0)]
        for kernel in pieces:
            graph.add_kernel(kernel)
            placement.set_kernel_device(kernel, hardware.devices[0])
            if kernels:
                graph.add_control_edge(kernels[-1], kernel)
            kernels.append(kernel)
    apply_layerwise_offload(graph, placement, hardware, 'toy',
                           prefetch_size=prefetch, layer_ids={0, 1, 2})
    graph.validate()
    placement.validate(graph)
    return graph, placement, hardware, kernels


@pytest.mark.parametrize(('prefetch', 'expected_us', 'copy_starts', 'peak'), [
    (0, 9, [0, 3, 6], 32_000),
    (1, 7, [0, 1, 3], 64_000),
])
def test_prefetch_timeline_and_bounded_hbm(prefetch, expected_us, copy_starts, peak):
    graph, placement, hardware, _ = toy(prefetch)
    result = Simulator(graph, placement, hardware).run()
    assert result.total_time_us == pytest.approx(expected_us)
    copies = sorted((e for e in result.trace if isinstance(e.kernel, WeightLoad)),
                    key=lambda e: e.start_us)
    assert [e.start_us for e in copies] == pytest.approx(copy_starts)
    assert result.peak_memory[hardware.find_local_memory(hardware.devices[0])] == peak
    assert result.peak_memory[hardware.host_memory] == 96_000
    stats = graph_stats(result, hardware)
    assert stats['host_to_device_bytes'] == 96_000
    assert stats['copy_time_ms'] == pytest.approx(.003)
    assert stats['exposed_transfer_time_ms'] == pytest.approx((expected_us - 6) / 1000)
    assert stats['fixed_weight_bytes'] == 0


def test_transfer_bound_pipeline_and_release():
    graph, placement, hardware, _ = toy(weight_bytes=96_000, compute_us=1)
    result = Simulator(graph, placement, hardware).run()
    assert result.total_time_us == pytest.approx(10)
    assert graph_stats(result, hardware)['exposed_transfer_time_ms'] == pytest.approx(.007)


def test_early_use_weights_live_until_end_of_layer():
    graph, placement, hardware, _ = toy(split=True)
    result = Simulator(graph, placement, hardware).run()
    # Weight read takes 0.02 us before each 2 us compute; it is not charged
    # again by the hold input, but its allocation persists to layer completion.
    assert result.total_time_us == pytest.approx(7.06)
    assert result.peak_memory[hardware.find_local_memory(hardware.devices[0])] == 64_000
    hardware.find_local_memory(hardware.devices[0]).capacity_gb = 63_999 / 1e9
    with pytest.raises(OOMError):
        Simulator(graph, placement, hardware).run()


@pytest.mark.parametrize('sp', [1, 2])
def test_official_components_have_native_precision_and_streaming(sp):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / f'examples/minimax_h3_ref2va_910b_sp{sp}.json').read_text())
    hardware = Ascend910B3(sp)
    workflow = create_workflow(hardware, config)
    results = workflow.run()
    expected = {
        'transformer_ref': ('bf16', 50),
        'transformer_text_refiner': ('bf16', 2),
        'qwen_vision': ('bf16', 27),
        'qwen_text': ('bf16', 50),
        'video_vae_encoder': ('fp32', 0),
        'audio_vae_encoder': ('fp32', 0),
        'audio_vae_decoder': ('fp32', 0),
        'video_vae_decoder': ('fp16', 36),
    }
    seen = set()
    for stage in results.values():
        for graph in stage.graphs:
            kernels = {e.kernel for e in graph.result.trace}
            component = next(iter(kernels)).csv_metadata['component']
            dtype, layer_count = expected[component]
            seen.add(component)
            assert {k.operator.compute_dtype for k in kernels if hasattr(k, 'operator')} == {dtype}
            stats = graph_stats(graph.result, hardware)
            assert stats['offloaded_layer_count'] == layer_count, component
            assert (stats['host_to_device_bytes'] > 0) == (layer_count > 0), component
    assert seen == set(expected)
    assert results['transformer_denoise'].graphs[0].multiplier == 49
    assert results['transformer_text_refiner'].graphs[0].multiplier == 1
    assert workflow.metadata['fits_peak_stage_memory']
