"""Model coefficients preserve the old effective-link contention behavior."""
import pytest

from rooflang.language.graph import ComputeGraph
from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.comm import AllGather
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.programs.models.minimax_h3.hardware import A5PRServer
from rooflang.programs.presets.ascend910b import Ascend910B3
from rooflang.runtime.efficiency import prepare_hardware, resolve_efficiency
from rooflang.runtime.simulator import Simulator


class _LegacyA5(A5PRServer):
    def find_aggregate_bandwidth(self, devices):
        return super().find_aggregate_bandwidth(devices) * 0.7 * 0.142857


def _run_concurrent_transfers(hardware):
    devices = sorted((node for node in hardware.nodes
                      if isinstance(node, Compute) and node.kind == 'gpu'),
                     key=lambda node: node.name)
    memories = [hardware.find_local_memory(device) for device in devices]
    graph = ComputeGraph()
    placement = Placement(hardware, graph)
    for stream, size in ((1, 4_000_000), (2, 6_000_000)):
        ready = Kernel()
        ready.tag = f'ready-{stream}'
        collective = AllGather(size, world=2)
        collective.tag = f'collective-{stream}'
        for rank, memory in enumerate(memories):
            for ports in (collective.inputs, collective.outputs):
                tensor = Tensor('fp16', (0,))
                ports[str(rank)] = tensor
                placement.set_tensor_memory(tensor, memory)
        graph.add_kernel(ready)
        graph.add_kernel(collective)
        graph.add_control_edge(ready, collective)
        placement.set_kernel_device(ready, devices[0], stream=stream)
    remote = Kernel(inputs={'x': Tensor('fp16', (2_000_000,))})
    remote.tag = 'remote-read'
    graph.add_kernel(remote)
    placement.set_tensor_memory(remote.inputs['x'], memories[1])
    placement.set_kernel_device(remote, devices[0], stream=3)
    graph.validate()
    placement.validate(graph)
    return Simulator(graph, placement, hardware).run()


@pytest.mark.parametrize('hardware_type', [Ascend910B3, A5PRServer])
def test_collectives_and_remote_buffer_preserve_effective_link_contention(hardware_type):
    legacy = (_LegacyA5(2) if hardware_type is A5PRServer else Ascend910B3(2))
    for _, _, data in legacy._graph.edges(data=True):
        for edge in data['fabrics']:
            if edge.src.kind == edge.dst.kind == 'gpu':
                edge.src_to_dst_bandwidth_gbs *= 0.7
                edge.dst_to_src_bandwidth_gbs *= 0.7
    legacy._clear_lookup_caches()
    values = {'intra_node_efficiency': 0.7}
    if hardware_type is A5PRServer:
        values['collective_efficiencies'] = {'2': 0.142857}
    prepared = prepare_hardware(hardware_type(2), resolve_efficiency(values))
    before = _run_concurrent_transfers(legacy)
    after = _run_concurrent_transfers(prepared)
    assert after.total_time_us == pytest.approx(before.total_time_us, rel=1e-12)
    expected = {(entry.kernel.tag, entry.device.name): entry for entry in before.trace}
    for entry in after.trace:
        previous = expected[entry.kernel.tag, entry.device.name]
        assert (entry.start_us, entry.end_us, entry.network_time_us,
                entry.network_elapsed_time_us) == pytest.approx(
                    (previous.start_us, previous.end_us, previous.network_time_us,
                     previous.network_elapsed_time_us), rel=1e-12)
        if not entry.kernel.tag.startswith('ready'):
            assert entry.start_us == 0.0
            assert entry.network_elapsed_time_us > entry.network_time_us
