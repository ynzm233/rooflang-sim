"""Graph rewrites must retain scheduling barriers or fail before mutation."""

import networkx as nx
import pytest

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.comm import Broadcast
from rooflang.language.kernels.forward import Nop
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.runtime.simulator import Simulator


def _tensor():
    return Tensor("bf16", (1,))


class Timed(Kernel):
    def __init__(self, time, **kwargs):
        super().__init__(**kwargs)
        self.time = time

    def scheduled_compute_time_us(self, device):
        return self.time


def test_remove_identity_keeps_barrier_timing_and_control_successors():
    hardware = HardwareGraph()
    devices = [Compute(str(i), {"bf16": 1}) for i in range(2)]
    memories = [Memory(f"m{i}", 1) for i in range(2)]
    for device, memory in zip(devices, memories):
        hardware.add_node(device)
        hardware.add_node(memory)
        hardware.add_edge(FabricEdge(device.name, device, memory, 1000, 1000, False))
    source = Timed(1, outputs={"x": _tensor()})
    identity = Nop(inputs={"x": _tensor()}, outputs={"y": _tensor()})
    sink = Timed(1, inputs={"y": _tensor()})
    barrier, follower = Timed(100), Timed(1)
    graph = ComputeGraph()
    for kernel in (source, identity, sink, barrier, follower):
        graph.add_kernel(kernel)
    graph.add_data_edge(source, identity, {"x": "x"})
    graph.add_data_edge(identity, sink, {"y": "y"})
    graph.add_control_edge(barrier, identity)
    graph.add_control_edge(identity, follower)
    placement = Placement(hardware=hardware, graph=graph)
    for kernel in (source, sink):
        placement.set_kernel_device(kernel, devices[0])
    for kernel in (barrier, follower):
        placement.set_kernel_device(kernel, devices[1])
    for tensor in (*identity.inputs.values(), *identity.outputs.values()):
        placement.set_tensor_memory(tensor, memories[0])
    before = Simulator(graph, placement, hardware).run()
    graph.remove_identity(identity)
    graph.validate()
    placement.validate(graph)
    after = Simulator(graph, placement, hardware).run()
    for kernel in (sink, follower):
        old = next(entry for entry in before.trace if entry.kernel is kernel)
        new = next(entry for entry in after.trace if entry.kernel is kernel)
        assert old.start_us == new.start_us == 100
    assert before.total_time_us == after.total_time_us == 101


def test_fusion_keeps_external_controls_without_a_self_dependency():
    graph = ComputeGraph()
    before, first, second, after = [Kernel() for _ in range(4)]
    for kernel in (before, first, second, after):
        graph.add_kernel(kernel)
    graph.add_control_edge(before, first)
    graph.add_control_edge(first, second)
    graph.add_control_edge(second, after)
    fused = graph.fuse_kernels(lambda kernels: Kernel(), [first, second])
    graph.validate()
    assert nx.has_path(graph._dag, before, fused)
    assert nx.has_path(graph._dag, fused, after)


@pytest.mark.parametrize("data_path", [False, True])
def test_fusion_rejects_external_dependency_cycle_before_mutation(data_path):
    graph = ComputeGraph()
    first = Kernel(outputs={"y": _tensor()})
    outside = Kernel(inputs={"x": _tensor()}, outputs={"y": _tensor()})
    second = Kernel(inputs={"x": _tensor()})
    for kernel in (first, outside, second):
        graph.add_kernel(kernel)
    for source, destination in ((first, outside), (outside, second)):
        if data_path:
            graph.add_data_edge(source, destination, {"y": "x"})
        else:
            graph.add_control_edge(source, destination)
    snapshot = graph._dag.copy()

    def callback(*args):
        pytest.fail("A rejected fusion must not invoke its callback")

    with pytest.raises(ValueError, match="external dependency path"):
        graph.fuse_kernels(callback, [first, second])
    assert nx.utils.graphs_equal(graph._dag, snapshot)


@pytest.mark.parametrize("terminal", [False, True])
def test_split_keeps_barriers_on_every_copy_and_completion(terminal):
    graph = ComputeGraph()
    before, after = Kernel(), Kernel()
    original = Kernel(inputs={"x": _tensor()},
                      outputs={} if terminal else {"y": _tensor()})
    for kernel in (before, original, after):
        graph.add_kernel(kernel)
    graph.add_control_edge(before, original)
    graph.add_control_edge(original, after)

    def split(kernel, count):
        fan = Nop(inputs={"x": _tensor()},
                  outputs={f"o{i}": _tensor() for i in range(count)})
        copies = [Kernel(inputs={"x": _tensor()},
                         outputs={} if terminal else {"y": _tensor()})
                  for _ in range(count)]
        outputs = {} if terminal else {"y": Nop(
            inputs={f"i{i}": _tensor() for i in range(count)}, outputs={"y": _tensor()})}
        return {"x": fan}, copies, outputs

    incoming, copies, outgoing = graph.split_kernel(split, original, 2)
    graph.validate()
    for kernel in (*incoming.values(), *copies, *outgoing.values()):
        assert nx.has_path(graph._dag, before, kernel)
        assert nx.has_path(graph._dag, kernel, after)


def test_split_rejects_unsupported_portless_split_without_losing_controls():
    graph = ComputeGraph()
    before, original, after = Kernel(), Kernel(), Kernel()
    for kernel in (before, original, after):
        graph.add_kernel(kernel)
    graph.add_control_edge(before, original)
    graph.add_control_edge(original, after)
    snapshot = graph._dag.copy()
    with pytest.raises(ValueError, match="non-empty prev_comms"):
        graph.split_kernel(lambda kernel, count: ({}, [Kernel(), Kernel()], {}), original, 2)
    assert nx.utils.graphs_equal(graph._dag, snapshot)


@pytest.mark.parametrize("operation", ["dup", "dedup"])
@pytest.mark.parametrize("target_broadcast", [False, True])
@pytest.mark.parametrize("incoming", [False, True])
def test_dup_and_dedup_reject_controls_before_callback_or_graph_mutation(
        operation, target_broadcast, incoming):
    graph = ComputeGraph()
    control = Kernel()
    broadcast = Broadcast(2, 2)
    broadcast.inputs = {"x": _tensor()}
    broadcast.outputs = {f"o{i}": _tensor() for i in range(2)}
    first = Kernel(inputs={"x": _tensor()}, outputs={"y": _tensor()})
    second = Kernel(inputs={"x": _tensor()}, outputs={"y": _tensor()})
    for kernel in (control, broadcast, first, second):
        graph.add_kernel(kernel)
    if operation == "dup":
        graph.add_data_edge(first, broadcast, {"y": "x"})
    else:
        graph.add_data_edge(broadcast, first, {"o0": "x"})
        graph.add_data_edge(broadcast, second, {"o1": "x"})
    target = broadcast if target_broadcast else first
    graph.add_control_edge(control if incoming else target,
                           target if incoming else control)
    snapshot = graph._dag.copy()

    def callback(*args):
        pytest.fail("A rejected rewrite must not invoke its callback")

    with pytest.raises(ValueError, match="control edges"):
        getattr(graph, operation)(callback, first if operation == "dup" else [first, second])
    assert nx.utils.graphs_equal(graph._dag, snapshot)
