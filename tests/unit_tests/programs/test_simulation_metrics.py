# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Configured shared simulations export metrics without model-specific code."""

import csv

import pytest

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.programs.experiments.simulation import simulate


class WorkloadKernel(Kernel):
    @property
    def flops(self):
        return self.inputs["x"].shape[0] * 1e6


def _workload(batch_size):
    gpu = Compute("gpu0", tflops={"bf16": 1.0})
    memory = Memory("hbm", capacity_gb=1)
    hardware = HardwareGraph()
    hardware.add_node(gpu)
    hardware.add_node(memory)
    hardware.add_edge(FabricEdge(
        "hbm_link", gpu, memory,
        src_to_dst_bandwidth_gbs=1000, dst_to_src_bandwidth_gbs=1000,
        is_full_duplex=False))
    graph = ComputeGraph()
    kernel = WorkloadKernel(inputs={"x": Tensor("bf16", (batch_size,))})
    graph.add_kernel(kernel)
    placement = Placement(hardware=hardware)
    placement.set_kernel_device(kernel, gpu)
    return graph, placement, hardware, kernel


def _row(path):
    with path.open(newline="", encoding="utf-8") as file:
        row, = csv.DictReader(file)
    return row


@pytest.mark.parametrize("model_name,stage_name,batch_size", [
    ("configured_language_model", "prefill", 2),
    ("configured_audio_model", "audio_decode", 5),
])
def test_shared_simulation_exports_configured_model_and_stage(
        tmp_path, model_name, stage_name, batch_size):
    graph, placement, hardware, kernel = _workload(batch_size)
    output_dir = tmp_path / model_name
    result = simulate(
        graph, placement, hardware, output_dir=output_dir,
        model_name=model_name, stage_name=stage_name,
        hardware_name="configured_chip", metadata={"batch_size": batch_size})

    assert {path.name for path in output_dir.iterdir()} == {
        f"{stage_name}.csv", "stage_metrics.csv",
        f"{model_name}_plan_metrics.csv"}
    operator = _row(output_dir / f"{stage_name}.csv")
    stage = _row(output_dir / "stage_metrics.csv")
    plan = _row(output_dir / f"{model_name}_plan_metrics.csv")
    assert result.trace[0].kernel is kernel
    assert result.total_time_us == pytest.approx(batch_size)
    assert operator["model_name"] == model_name
    assert operator["hardware_name"] == "configured_chip"
    assert float(operator["flops"]) == batch_size * 1e6
    assert stage["name"] == stage_name
    assert plan["stages"] == stage_name
    assert plan["batch_size"] == str(batch_size)
    assert float(plan["execution_time_ms"]) == pytest.approx(batch_size / 1000)


def test_measurement_start_remains_supported_without_file_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph, placement, hardware, kernel = _workload(3)
    result = simulate(graph, placement, hardware, measurement_start=kernel)

    assert result.trace[0].kernel is kernel
    assert result.measured_time_us == pytest.approx(3)
    assert list(tmp_path.iterdir()) == []


def test_csv_export_uses_hardware_name_without_writing_timeline(tmp_path):
    graph, placement, hardware, _ = _workload(1)
    output_dir = tmp_path / "metrics"
    simulate(graph, placement, hardware,
             output_dir=output_dir, model_name="custom", stage_name="forward")

    assert not list(tmp_path.rglob("*.json"))
    assert _row(output_dir / "custom_plan_metrics.csv")["chip_name"] == "HardwareGraph"


@pytest.mark.parametrize("identities", [{}, {"model_name": "model"},
                                        {"stage_name": "forward"}])
def test_csv_export_requires_explicit_model_and_stage_before_simulation(
        tmp_path, identities):
    output_dir = tmp_path / "metrics"
    with pytest.raises(ValueError, match="requires model_name and stage_name"):
        simulate(None, None, None, output_dir=output_dir, **identities)
    assert not output_dir.exists()


def test_default_simulation_returns_result_without_writing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph, placement, hardware, _ = _workload(1)
    result = simulate(graph, placement, hardware)
    assert result.trace
    assert list(tmp_path.iterdir()) == []
