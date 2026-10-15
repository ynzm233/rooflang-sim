"""Legacy roofline attribution stays distinct from DES elapsed time."""

import csv

import pytest

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.comm import AllReduce
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.runtime.export_config import csv_fields
from rooflang.runtime.metrics_export import export_metrics
from rooflang.runtime.simulator import Bound, SimulationResult, Simulator, TraceEntry
from rooflang.runtime.stages import GraphResult, StageResult


class WorkKernel(Kernel):
    def __init__(self, compute_us, memory_us):
        super().__init__(inputs={"x": Tensor("bf16", (int(memory_us * 500),))})
        self._flops = compute_us * 1e6

    @property
    def flops(self):
        return self._flops


def _simulate(compute_us, memory_us, *, parallel=False):
    gpu = Compute("gpu", {"bf16": 1.0}, kind="gpu")
    hbm = Memory("hbm", capacity_gb=1)
    hardware = HardwareGraph()
    hardware.add_node(gpu)
    hardware.add_node(hbm)
    hardware.add_edge(FabricEdge(
        "hbm_link", gpu, hbm, src_to_dst_bandwidth_gbs=1.0,
        dst_to_src_bandwidth_gbs=1.0, is_full_duplex=False))
    graph = ComputeGraph()
    placement = Placement(hardware=hardware)
    for stream in range(2 if parallel else 1):
        kernel = WorkKernel(compute_us, memory_us)
        graph.add_kernel(kernel)
        placement.set_kernel_device(kernel, gpu, stream=stream)
    return Simulator(graph, placement, hardware).run()


def _read(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(reader)


@pytest.mark.parametrize("model_name", ["magi2_preview", "dsv4_pro"])
@pytest.mark.parametrize("compute,memory,expected", [
    (2, 1, (0.002, 0)),
    (1, 2, (0, 0.002)),
    (2, 2, (0.002, 0.002)),
])
def test_roofline_attribution_uses_dominant_ideal_component(
    tmp_path, model_name, compute, memory, expected,
):
    result = _simulate(compute, memory)
    export_metrics({"forward": result}, tmp_path, model_name=model_name)

    fields, (row,) = _read(tmp_path / "forward.csv")
    assert float(row["overlapped_compute_ms"]) == pytest.approx(expected[0])
    assert float(row["overlapped_memory_ms"]) == pytest.approx(expected[1])
    assert float(row["execution_time_ms"]) == pytest.approx(max(compute, memory) / 1000)
    # Dtype throughput does not provide independent cube/vector engine times.
    assert "cube_compute_ms" not in fields
    assert "vector_compute_ms" not in fields


def test_overlap_attribution_does_not_absorb_resource_contention(tmp_path):
    result = _simulate(2, 1, parallel=True)
    export_metrics({"forward": result}, tmp_path, model_name="dsv4_pro")

    _, rows = _read(tmp_path / "forward.csv")
    assert result.total_time_us == pytest.approx(4)
    for row in rows:
        assert float(row["execution_time_ms"]) == pytest.approx(0.004)
        assert float(row["compute_ms"]) == pytest.approx(0.002)
        assert float(row["overlapped_compute_ms"]) == pytest.approx(0.002)
        assert float(row["overlapped_memory_ms"]) == 0


def _communication_result():
    kernel = AllReduce(1024, 2)
    device = Compute("gpu", {"bf16": 1.0}, kind="gpu")
    entry = TraceEntry(
        kernel, device, 0, 0, 1000, Bound.NETWORK,
        compute_time_us=0, memory_time_us=0, network_time_us=1000,
        local_elapsed_time_us=0, network_elapsed_time_us=1000,
    )
    return SimulationResult([entry], 1000, {})


def test_communication_only_stage_does_not_invent_overlap_columns(tmp_path):
    export_metrics({"broadcast": _communication_result()}, tmp_path,
                   model_name="magi2_preview")

    fields, (row,) = _read(tmp_path / "broadcast.csv")
    assert "overlapped_compute_ms" not in fields
    assert "overlapped_memory_ms" not in fields
    assert float(row["network_ms"]) == 1


def test_communication_and_unmodeled_cpu_leave_overlap_cells_blank(tmp_path):
    stage = StageResult(
        graphs=[GraphResult("compute", _simulate(2, 1)),
                GraphResult("broadcast", _communication_result())],
        untimed_rows=[{"node_name": "encoder", "device": "cpu", "cpu_flops": 123}],
    )
    export_metrics({"text_encode": stage}, tmp_path, model_name="magi2_preview")

    _, rows = _read(tmp_path / "text_encode.csv")
    assert float(rows[0]["overlapped_compute_ms"]) == pytest.approx(0.002)
    for row in rows[1:]:
        assert row["overlapped_compute_ms"] == ""
        assert row["overlapped_memory_ms"] == ""
    assert rows[-1]["execution_time_ms"] == ""


@pytest.mark.parametrize("model_name", ["magi2_preview", "dsv4_pro"])
def test_optional_timing_fields_require_supplied_values(model_name):
    fields = ("overlapped_compute_ms", "overlapped_memory_ms")
    absent = csv_fields("operator", model_name, [dict.fromkeys(fields)])
    assert not set(fields).intersection(absent)
    present = csv_fields("operator", model_name, [dict.fromkeys(fields, 0.0)])
    assert set(fields).issubset(present)
