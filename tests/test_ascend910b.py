"""Hand-calculated checks for B3 compute, memory, host transfer and CLI."""
import json
from types import SimpleNamespace

import pytest

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.programs.experiments import main as main_module
from rooflang.programs.models.minimax_h3.bridge import H3Collective
from rooflang.programs.models.minimax_h3.workflow import create_workflow
from rooflang.programs.presets.ascend910b import Ascend910B, Ascend910B3
from rooflang.runtime.reproducibility import config_snapshot
from rooflang.runtime.simulator import OOMError, Simulator


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
    placement = Placement(hardware)
    device = next(node for node in hardware.nodes if node.kind == "gpu")
    placement.set_kernel_device(kernel, device)
    return Simulator(graph, placement, hardware).run()


@pytest.mark.parametrize(("segments", "expected_us"), [
    ({"fp16:cube": 294_912_000, "fp16:vector": 36_864_000}, 3.0),
    ({"bf16:cube": 294_912_000, "bf16:vector": 36_864_000}, 3.0),
    ({"fp32:cube": 73_728_000, "fp32:vector": 18_432_000}, 3.0),
])
def test_cube_and_vector_peaks_drive_separate_costs(segments, expected_us):
    result = run_single(Ascend910B(), ComputeSegments(segments))
    assert result.total_time_us == pytest.approx(expected_us / 0.7)
    with pytest.raises(ValueError, match="no compute for dtype 'fp8:cube'"):
        run_single(Ascend910B(), ComputeSegments({"fp8:cube": 1_000_000}))


def test_hbm_bandwidth_and_physical_capacity_units():
    hardware = Ascend910B()
    traffic = ComputeSegments(inputs={"x": Tensor("fp16", (800_000,))},
                              outputs={"y": Tensor("fp16", (1_600_000,))})
    # 1.6 MB read + 3.2 MB write at decimal 1600 GB/s = 3 us.
    assert run_single(hardware, traffic).total_time_us == pytest.approx(3.0)
    capacity = 64 * 1024**3
    fits = ComputeSegments(inputs={"x": Tensor("fp16", (capacity // 2,))})
    result = run_single(hardware, fits)
    assert max(result.peak_memory.values()) == capacity
    too_large = ComputeSegments(inputs={"x": Tensor("fp16", (capacity // 2 + 1,))})
    with pytest.raises(OOMError) as error:
        run_single(hardware, too_large)
    assert error.value.capacity_bytes == capacity
    assert error.value.used_bytes == capacity + 2


@pytest.mark.parametrize(("efficiency", "expected_us"), [
    (0.6, 5.0), (0.3, 10.0),
])
def test_local_memory_efficiency_changes_time_without_inflating_bytes(
    efficiency, expected_us,
):
    hardware = Ascend910B3()
    assumptions = hardware.metadata["simulation_assumptions"]
    assert assumptions["cube_memory_efficiency"] == assumptions["vector_memory_efficiency"] == 0.6
    kernel = ComputeSegments(inputs={"x": Tensor("fp16", (800_000,))},
                             outputs={"y": Tensor("fp16", (1_600_000,))})
    physical_result = run_single(hardware, kernel)
    kernel.local_memory_efficiency = efficiency
    result = run_single(hardware, kernel)
    assert result.total_time_us == pytest.approx(expected_us)
    assert result.trace[0].memory_time_us == pytest.approx(expected_us)
    assert kernel.transferred_bytes == 4_800_000
    assert result.peak_memory == physical_result.peak_memory
    assert kernel.to_dict()["local_memory_efficiency"] == efficiency
    fabric = hardware.find_fabric(hardware.devices[0], hardware.find_local_memory(hardware.devices[0]))
    assert fabric.src_to_dst_bandwidth_gbs == fabric.dst_to_src_bandwidth_gbs == 1600


@pytest.mark.parametrize("efficiency", [0, -0.1, 1.1, float("nan"), float("inf"), True, None])
def test_invalid_local_memory_efficiency_is_rejected(efficiency):
    kernel = ComputeSegments()
    kernel.local_memory_efficiency = efficiency
    with pytest.raises(ValueError, match="local_memory_efficiency"):
        run_single(Ascend910B3(), kernel)


def test_two_card_collective_uses_direct_per_direction_bandwidth():
    hardware = Ascend910B(2)
    devices = sorted((node for node in hardware.nodes if node.kind == "gpu"),
                     key=lambda node: node.name)
    fabric = hardware.find_fabric(*devices)
    assert fabric.is_full_duplex
    assert fabric.src_to_dst_bandwidth_gbs == fabric.dst_to_src_bandwidth_gbs == 28.0
    assert hardware.find_aggregate_bandwidth(devices) == 28.0
    collective = H3Collective(payload=560_000, world=2)
    graph = ComputeGraph()
    graph.add_kernel(collective)
    placement = Placement(hardware)
    for rank, device in enumerate(devices):
        for ports in (collective.inputs, collective.outputs):
            tensor = Tensor("fp16", (0,))
            ports[str(rank)] = tensor
            placement.set_tensor_memory(tensor, hardware.find_local_memory(device))
    # H3 sends half the payload per rank: 280 kB / 28 GB/s = 10 us.
    result = Simulator(graph, placement, hardware).run()
    assert result.total_time_us == pytest.approx(10.0)


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
@pytest.mark.parametrize("memory_efficiency", [1.0, 0.3])
def test_host_transfer_uses_host_link_and_keeps_hbm_local(direction, memory_efficiency):
    hardware = Ascend910B3(2)
    device = hardware.dma_devices[0]
    host_tensor = Tensor("fp16", (160_000,))
    hbm_tensor = Tensor("fp16", (160_000,))
    ports = {
        "inputs": {"source": host_tensor if direction == "h2d" else hbm_tensor},
        "outputs": {"destination": hbm_tensor if direction == "h2d" else host_tensor},
    }
    kernel = ComputeSegments(**ports)
    assert kernel.local_memory_efficiency == 1.0
    kernel.local_memory_efficiency = memory_efficiency
    graph = ComputeGraph()
    graph.add_kernel(kernel)
    placement = Placement(hardware)
    placement.set_kernel_device(kernel, device)
    placement.set_tensor_memory(host_tensor, hardware.host_memory)
    assert hardware.find_local_memory(device).kind == "hbm"
    assert hardware.find_local_memory(device) is hardware.find_local_memory(hardware.devices[0])
    fabric = hardware.find_fabric(hardware.host_memory, device)
    assert fabric.src_to_dst_bandwidth_gbs == fabric.dst_to_src_bandwidth_gbs == 32.0
    assert fabric.is_full_duplex
    assert hardware.host_memory.capacity_gb * 1e9 == 1536 * 1024**3
    # 320 kB in either direction / 32 GB/s = 10 us.
    result = Simulator(graph, placement, hardware).run()
    assert result.total_time_us == pytest.approx(10.0)
    assert result.trace[0].memory_time_us == pytest.approx(0.2 / memory_efficiency)
    assert result.trace[0].network_time_us == pytest.approx(10.0)
    assert kernel.transferred_bytes == 640_000


def test_dma_prefetch_overlaps_independent_npu_compute():
    hardware = Ascend910B3()
    host_tensor = Tensor("fp16", (160_000,))
    copy = ComputeSegments(inputs={"source": host_tensor},
                           outputs={"destination": Tensor("fp16", (160_000,))})
    compute = ComputeSegments({"fp16:cube": 2_064_384_000})
    graph = ComputeGraph()
    graph.add_kernel(copy)
    graph.add_kernel(compute)
    placement = Placement(hardware)
    placement.set_kernel_device(copy, hardware.dma_devices[0])
    placement.set_kernel_device(compute, hardware.devices[0])
    placement.set_tensor_memory(host_tensor, hardware.host_memory)
    # Both independent tasks take 10 us and occupy different modeled engines.
    result = Simulator(graph, placement, hardware).run()
    assert result.total_time_us == pytest.approx(10.0)


@pytest.mark.parametrize(("hardware_name", "cards"), [
    ("Ascend910B3_1Card", 1), ("Ascend910B3_2Cards", 2),
    ("Ascend910B1Card", 1), ("Ascend910B2Cards", 2),
])
def test_cli_selects_matching_hardware_and_h3_sequence_parallelism(
    hardware_name, cards, tmp_path, monkeypatch,
):
    config = {"workflow": "ref2va", "sp": cards}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    seen = []

    def capture(workflow, **kwargs):
        hardware = kwargs["hardware"]
        assert isinstance(hardware, Ascend910B3)
        assert hardware.n_devices == cards
        assert workflow.metadata["main_sp"] == cards
        assert kwargs["hardware_name"] == hardware_name
        seen.append(kwargs["config"])
        return {}, tmp_path

    monkeypatch.setattr(main_module, "run_and_export", capture)
    main_module.main(["--model", "minimax_h3", "--hardware", hardware_name,
                      "--config", str(path), "--output-dir", str(tmp_path)])
    assert seen == [config]
    if cards == 1:
        with pytest.raises(ValueError, match="SP=2 requires at least 2 devices"):
            create_workflow(Ascend910B(), {"sp": 2})


def test_snapshot_preserves_sources_and_uncalibrated_capacity_assumptions():
    hardware = Ascend910B(2)
    snapshot = config_snapshot(
        SimpleNamespace(metadata={}), model_name="minimax_h3",
        hardware_name="Ascend910B3_2Cards", stage=None, config={}, hardware=hardware,
    )
    saved = json.loads(json.dumps(snapshot))["hardware"]["metadata"]
    assert saved["official_specification"]["product"] == "Ascend 910B3"
    assert saved["official_specification"]["reported_fp16_total_tflops"] == 313.0
    assert saved["derived_specification"]["cube_tflops"]["fp16"] == 294.912
    assert saved["derived_specification"]["vector_tflops"]["fp16"] == 18.432
    assert Ascend910B is Ascend910B3
    assumptions = saved["simulation_assumptions"]
    assert assumptions["cube_efficiency"] == assumptions["vector_efficiency"] == 0.7
    assert assumptions["cube_memory_efficiency"] == 0.6
    assert assumptions["vector_memory_efficiency"] == 0.6
    assert assumptions["memory_efficiency"] == 1.0
    assert assumptions["memory_capacity_gb"] == 68.719476736
    assert assumptions["driver_runtime_reserved_bytes"] == 0
    assert assumptions["intra_node_bandwidth_gbs"] == 28.0
    assert assumptions["host_bandwidth_gbs_per_device_per_direction"] == 32.0
    assert not assumptions["host_shared_bandwidth_contention_modeled"]
    assert "uncalibrated" in assumptions["calibration"]
    assumptions["memory_capacity_source_urls"].clear()
    assert hardware.metadata["simulation_assumptions"]["memory_capacity_source_urls"]
    for invalid in (True, 0, 3, 1.0):
        with pytest.raises(ValueError, match="1 or 2 devices"):
            Ascend910B(invalid)
