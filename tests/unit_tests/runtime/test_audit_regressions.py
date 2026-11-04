"""Small, end-to-end regressions for simulator audit findings."""

from fractions import Fraction

import pytest

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.comm import Broadcast
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.runtime.simulator import OOMError, Simulator


def _broadcast(*, destination_capacity=2000, storage_bytes=None, inplace=False, root=False):
    hardware = HardwareGraph()
    devices = [Compute(f"gpu{i}", {"bf16": 1}, kind="gpu") for i in range(2)]
    memories = [Memory("hbm0", 2000 / 1e9),
                Memory("hbm1", destination_capacity / 1e9)]
    for device, memory in zip(devices, memories):
        hardware.add_node(device)
        hardware.add_node(memory)
        hardware.add_edge(FabricEdge(device.name, device, memory, 1000, 1000, False))
    hardware.add_edge(FabricEdge("link", *devices, 10, 10, True))
    source = Kernel(outputs={"y": Tensor("bf16", (500,))})
    broadcast = Broadcast(1000, 2)
    broadcast.inputs = {"x": source.outputs["y"]}
    broadcast.outputs = {
        str(i): Tensor("bf16", (500,), storage_bytes=storage_bytes) for i in range(2)
    }
    if inplace:
        broadcast.outputs["0"] = source.outputs["y"]
    graph = ComputeGraph()
    graph.add_kernel(broadcast)
    if not root:
        graph.add_kernel(source)
        graph.add_data_edge(source, broadcast, {"y": "x"})
    placement = Placement(hardware=hardware, graph=graph)
    if not root:
        placement.set_kernel_device(source, devices[0])
    placement.set_tensor_memory(broadcast.inputs["x"], memories[0])
    for i, memory in enumerate(memories):
        placement.set_tensor_memory(broadcast.outputs[str(i)], memory)
    graph.validate()
    placement.validate(graph)
    return Simulator(graph, placement, hardware), broadcast, memories


def test_broadcast_receive_buffer_exceeding_destination_capacity_raises():
    simulator, broadcast, memories = _broadcast(destination_capacity=1)
    with pytest.raises(OOMError) as error:
        simulator.run()
    assert error.value.memory is memories[1]
    assert error.value.trigger_kernel is broadcast
    assert error.value.used_bytes == 1000


@pytest.mark.parametrize("storage_bytes,inplace,expected", [
    (None, False, (2000, 1000)),
    (None, True, (1000, 1000)),
    (0, False, (1000, 0)),
])
def test_broadcast_uses_declared_storage_and_explicit_tensor_aliases(
        storage_bytes, inplace, expected):
    simulator, _, memories = _broadcast(storage_bytes=storage_bytes, inplace=inplace)
    result = simulator.run()
    assert tuple(result.peak_memory.get(memory, 0) for memory in memories) == expected
    assert result.total_time_us == pytest.approx(0.101)


def test_root_broadcast_keeps_inplace_input_as_its_output_storage():
    simulator, _, memories = _broadcast(inplace=True, root=True)
    memories[0].capacity_gb = 1000 / 1e9
    result = simulator.run()
    assert tuple(result.peak_memory[memory] for memory in memories) == (1000, 1000)
    assert simulator._mem_usage[memories[0]] == 1000


@pytest.mark.parametrize("reverse", [False, True])
def test_local_memory_timing_respects_fabric_endpoint_order(reverse):
    hardware = HardwareGraph()
    device = Compute("gpu", {"bf16": 1})
    memory = Memory("hbm", 1)
    hardware.add_node(device)
    hardware.add_node(memory)
    hardware.add_edge(FabricEdge(
        "hbm_link", memory if reverse else device, device if reverse else memory,
        1 if reverse else 100, 100 if reverse else 1, False))
    kernel = Kernel(inputs={"x": Tensor("bf16", (500_000,))},
                    outputs={"y": Tensor("bf16", (1_000_000,))})
    graph = ComputeGraph()
    graph.add_kernel(kernel)
    placement = Placement(hardware=hardware)
    placement.set_kernel_device(kernel, device)
    result = Simulator(graph, placement, hardware).run()
    assert result.total_time_us == pytest.approx(1020)


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), float("-inf"),
                                 0, -1, 2, True, "0.5", None])
def test_resource_cap_rejects_invalid_values_before_placing_kernel(cap):
    placement = Placement()
    kernel = Kernel()
    with pytest.raises(ValueError, match="resource_cap"):
        placement.set_kernel_device(kernel, Compute("gpu"), resource_cap=cap)
    assert not placement.placed_kernels


@pytest.mark.parametrize("cap", [1, 0.5, Fraction(1, 2)])
def test_resource_cap_accepts_finite_real_fractions(cap):
    placement = Placement()
    kernel = Kernel()
    placement.set_kernel_device(kernel, Compute("gpu"), resource_cap=cap)
    assert placement.get_kernel_device(kernel).resource_cap == cap
