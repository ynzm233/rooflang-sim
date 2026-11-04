"""Shared remote-memory links retain each transfer's efficiency."""

import pytest

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.runtime.efficiency import prepare_hardware
from rooflang.runtime.simulator import Simulator


def _hardware(intra_node_efficiency):
    hardware = HardwareGraph()
    switch = Compute("switch", kind="switch")
    hardware.add_node(switch)
    for name, node_id in (("g0", 0), ("g1", 0), ("g2", 1)):
        device = Compute(name, {"fp16": 1}, kind="gpu")
        device.node_id = node_id
        memory = Memory(f"{name}-hbm", capacity_gb=1, kind="hbm")
        hardware.add_node(device)
        hardware.add_node(memory)
        hardware.add_edge(FabricEdge(
            f"hbm-{name}", device, memory, 10000, 10000, False))
        hardware.add_edge(FabricEdge(
            f"net-{name}", device, switch, 100, 100, True))
    return prepare_hardware(hardware, {
        "intra_node_efficiency": intra_node_efficiency,
    })


@pytest.mark.parametrize("role", ["inputs", "outputs"])
@pytest.mark.parametrize("intra_node_efficiency", [1.0, 0.5])
def test_shared_remote_link_time_is_independent_of_tensor_order(
        role, intra_node_efficiency):
    hardware = _hardware(intra_node_efficiency)
    nodes = {node.name: node for node in hardware.nodes}
    times = []
    for order in (("g1", "g2"), ("g2", "g1")):
        tensors = {name: Tensor("fp16", (500,)) for name in order}
        kernel = Kernel(**{role: tensors})
        graph = ComputeGraph()
        graph.add_kernel(kernel)
        placement = Placement(hardware, graph)
        placement.set_kernel_device(kernel, nodes["g0"])
        for name, tensor in tensors.items():
            placement.set_tensor_memory(tensor, nodes[f"{name}-hbm"])
        result = Simulator(graph, placement, hardware).run()
        assert result.trace[0].network_time_us == result.total_time_us
        assert sum(tensor.size_bytes for tensor in tensors.values()) == 2000
        times.append(result.total_time_us)

    # The shared 100 GB/s edge serves 1000 bytes of each transfer; only g1
    # shares g0's node, so its service time gets the intra-node coefficient.
    expected_us = 1000 / (100 * 1e3 * intra_node_efficiency) + 1000 / (100 * 1e3)
    assert times == pytest.approx([expected_us, expected_us])
