"""Hand-calculated timing and capacity checks for the official 950PR profile."""
import json
from types import SimpleNamespace

import pytest

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.programs.models.minimax_h3.bridge import H3Collective
from rooflang.programs.models.minimax_h3.hardware import A5PRServer
from rooflang.runtime.reproducibility import config_snapshot
from rooflang.runtime.simulator import Bound, OOMError, Simulator


class ComputeSegments(Kernel):
    def __init__(self, segments=None, **kwargs):
        super().__init__(**kwargs)
        self.segments = segments or {}

    @property
    def flops_by_dtype(self):
        return self.segments

    @property
    def flops(self):
        return sum(self.segments.values())


def run_single(hardware, kernel):
    graph = ComputeGraph()
    graph.add_kernel(kernel)
    placement = Placement(hardware=hardware)
    device = next(node for node in hardware.nodes if node.kind == 'gpu')
    placement.set_kernel_device(kernel, device)
    return Simulator(graph, placement, hardware).run()


@pytest.mark.parametrize(('segments', 'expected_us'), [
    # FP16 Cube: 302.4 MFLOPs / 302.4 TFLOPS = 1 us;
    # Vector: 64.8 MFLOPs / 32.4 TFLOPS = 2 us. Segments add.
    ({'fp16:cube': 302_400_000, 'fp16:vector': 64_800_000}, 3.0),
    # FP8 Cube: 865 TFLOPS at 70% efficiency = 605.5 TFLOPS.
    ({'fp8:cube': 605_500_000}, 1.0),
])
def test_official_compute_peaks_drive_simulated_cost(segments, expected_us):
    result = run_single(A5PRServer(1), ComputeSegments(segments))
    assert result.total_time_us == pytest.approx(expected_us)
    assert result.trace[0].compute_time_us == pytest.approx(expected_us)
    assert result.trace[0].bound is Bound.COMPUTE


def test_unsupported_vector_fp8_cannot_silently_use_cube_throughput():
    with pytest.raises(ValueError, match="no compute for dtype 'fp8:vector'"):
        run_single(A5PRServer(1), ComputeSegments({'fp8:vector': 1_000_000}))


def test_hbm_traffic_uses_decimal_bandwidth():
    # Read 1.28 MB and write 2.56 MB at 1600 GB/s * 0.8: 3 us.
    kernel = ComputeSegments(
        inputs={'x': Tensor('fp16', (640_000,))},
        outputs={'y': Tensor('fp16', (1_280_000,))},
    )
    result = run_single(A5PRServer(1), kernel)
    assert result.total_time_us == pytest.approx(3.0)
    assert result.trace[0].memory_time_us == pytest.approx(3.0)
    assert result.trace[0].bound is Bound.MEMORY


def test_capacity_boundary_is_128_decimal_gb():
    hardware = A5PRServer(1)
    # These are shape-only tensors; no physical 128 GB allocation is made.
    fits = ComputeSegments(inputs={'x': Tensor('fp16', (64_000_000_000,))})
    result = run_single(hardware, fits)
    memory = next(node for node in hardware.nodes if node.kind == 'hbm')
    assert result.peak_memory[memory] == 128_000_000_000

    too_large = ComputeSegments(inputs={'x': Tensor('fp16', (64_000_000_001,))})
    with pytest.raises(OOMError) as error:
        run_single(hardware, too_large)
    assert error.value.capacity_bytes == 128_000_000_000
    assert error.value.used_bytes == 128_000_000_002


def test_h3_collective_retains_assumed_bandwidth_and_latency():
    hardware = A5PRServer(4)
    collective = H3Collective(payload=1_568_000, world=4)
    graph = ComputeGraph()
    graph.add_kernel(collective)
    placement = Placement(hardware=hardware)
    memories = sorted((node for node in hardware.nodes if node.kind == 'hbm'),
                      key=lambda node: node.name)
    for rank, memory in enumerate(memories):
        for ports in (collective.inputs, collective.outputs):
            tensor = Tensor('fp16', (0,))
            ports[str(rank)] = tensor
            placement.set_tensor_memory(tensor, memory)

    result = Simulator(graph, placement, hardware).run()
    # 3/4 of 1.568 MB traverses the modeled fabric. The retained rounded
    # 4-card factor gives 117.5998824 GB/s: 10.000010 us + 1 us latency.
    assert result.total_time_us == pytest.approx(11.00001000001)
    assert all(entry.bound is Bound.NETWORK for entry in result.trace)


def test_saved_configuration_distinguishes_specifications_and_assumptions():
    hardware = A5PRServer(1)
    snapshot = config_snapshot(
        SimpleNamespace(metadata={}), model_name='MiniMax-H3',
        hardware_name='A5_pr_server', stage=None, config={}, hardware=hardware,
    )
    # Exercise the same JSON representation saved to config.snapshot.json.
    saved = json.loads(json.dumps(snapshot))['hardware']['metadata']
    official = saved['official_specification']
    assumed = saved['simulation_assumptions']
    assert official['cube_tflops'] == {'fp16': 432.0, 'fp8': 865.0}
    assert official['vector_tflops'] == {'fp16': 54.0}
    assert official['source_url'].startswith('https://public-download.')
    assert assumed['cube_efficiency'] == 0.7
    assert assumed['intra_node_latency_us'] == 1.0
    assert assumed['collective_efficiencies']['4'] == 0.428571
    # Exported evidence must remain independent of later hardware edits.
    snapshot['hardware']['metadata']['simulation_assumptions']['cube_efficiency'] = 1.0
    assert hardware.metadata['simulation_assumptions']['cube_efficiency'] == 0.7
