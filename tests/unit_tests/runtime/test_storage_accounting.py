"""Aggregate traffic is not necessarily a simultaneously allocated tensor."""

import pytest

from rooflang.language.graph import ComputeGraph, HardwareGraph, FabricEdge
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.runtime.simulator import Simulator, OOMError


def test_traffic_storage_override_preserves_bandwidth_and_normal_allocations():
    gpu = Compute("gpu", {"bf16": 1}, kind="gpu")
    memory = Memory("hbm", capacity_gb=32 / 1e9)
    hardware = HardwareGraph()
    hardware.add_node(gpu)
    hardware.add_node(memory)
    hardware.add_edge(FabricEdge("hbm_link", gpu, memory, 1000, 1000, False))
    traffic = Tensor("bf16", (500_000_000,), storage_bytes=0)
    first = Kernel(outputs={"traffic": traffic},
                   weights={"w": Tensor("bf16", (4,))})
    second = Kernel(inputs={"traffic": traffic}, outputs={"out": Tensor("bf16", (4,))})
    graph = ComputeGraph()
    graph.add_kernel(first)
    graph.add_kernel(second)
    graph.add_data_edge(first, second, {"traffic": "traffic"})
    placement = Placement(hardware=hardware)
    placement.set_kernel_device(first, gpu)
    placement.set_kernel_device(second, gpu)
    result = Simulator(graph, placement, hardware).run()
    assert traffic.size_bytes == 1e9
    assert result.total_time_us == pytest.approx((2e9 + 16) / 1e6)
    assert result.peak_memory[memory] == 16
    traffic._storage_bytes = None
    with pytest.raises(OOMError):
        Simulator(graph, placement, hardware).run()


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf")])
def test_invalid_storage_override_rejected(amount):
    with pytest.raises(ValueError, match="storage_bytes"):
        Tensor("bf16", (1,), storage_bytes=amount)
