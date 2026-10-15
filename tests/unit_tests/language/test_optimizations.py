# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Unit tests for placement-independent communication graph optimization."""


import pytest

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.comm import AllGather, AllReduce, AllToAll, Broadcast, Gather, Reduce, ReduceScatter, Scatter, Send, Recv
from rooflang.language.tensor import Tensor
from rooflang.language.placement import Placement
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.optimization.comm import (
    _create_collective, optimize_comms,
)


SHARD = Tensor("bf16", (4, 4))


def _build_chain(collector, distributor, n, same_devices=True):
    """Build n preds → collector → distributor → n succs with placement."""
    g = ComputeGraph()
    gpus = [Compute(name=f"gpu{i}") for i in range(n)]

    preds = [Kernel(outputs={"y": Tensor("bf16", SHARD.shape)})
             for _ in range(n)]
    succs = [Kernel(inputs={"a": Tensor("bf16", SHARD.shape)})
             for _ in range(n)]

    collector.inputs = {
        f"x{i}": Tensor("bf16", SHARD.shape) for i in range(n)
    }
    collector.outputs = {"z": Tensor("bf16", SHARD.shape)}
    distributor.inputs = {"z": Tensor("bf16", SHARD.shape)}
    distributor.outputs = {
        f"y{i}": Tensor("bf16", SHARD.shape) for i in range(n)
    }

    for k in preds + [collector, distributor] + succs:
        g.add_kernel(k)
    for i, p in enumerate(preds):
        g.add_data_edge(p, collector, {"y": f"x{i}"})
    g.add_data_edge(collector, distributor, {"z": "z"})
    for i, s in enumerate(succs):
        g.add_data_edge(distributor, s, {f"y{i}": "a"})

    other_gpus = []
    if not same_devices:
        other_gpus = [Compute(name=f"other{i}") for i in range(n)]
    hw = HardwareGraph()
    memories = {}
    for device in gpus + other_gpus:
        memory = Memory(name=f"{device.name}-hbm", capacity_gb=1.0)
        hw.add_node(device)
        hw.add_node(memory)
        hw.add_edge(FabricEdge(
            name=f"{device.name}-hbm", src=device, dst=memory,
            src_to_dst_bandwidth_gbs=1.0,
            dst_to_src_bandwidth_gbs=1.0,
            is_full_duplex=False,
        ))
        memories[device] = memory

    placement = Placement(hardware=hw, graph=g)
    for i, p in enumerate(preds):
        placement.set_kernel_device(p, gpus[i])
    if same_devices:
        for i, s in enumerate(succs):
            placement.set_kernel_device(s, gpus[i])
        successor_devices = gpus
    else:
        for i, s in enumerate(succs):
            placement.set_kernel_device(s, other_gpus[i])
        successor_devices = other_gpus

    for i, tensor in enumerate(collector.inputs.values()):
        placement.set_tensor_memory(tensor, memories[gpus[i]])
    placement.set_tensor_memory(collector.outputs["z"], memories[gpus[0]])
    placement.set_tensor_memory(distributor.inputs["z"], memories[gpus[0]])
    for device, tensor in zip(
            successor_devices, distributor.outputs.values()):
        placement.set_tensor_memory(tensor, memories[device])

    return g, placement, preds, succs


class TestFuseWorld4:
    def test_reduce_broadcast_to_allreduce(self):
        r = Reduce(total_bytes=32, world=4, dtype="bf16")
        b = Broadcast(total_bytes=32, world=4)
        g, p, preds, succs = _build_chain(r, b, n=4)
        optimize_comms(g)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), AllReduce)

    def test_reduce_scatter_to_reducescatter(self):
        r = Reduce(total_bytes=32, world=4, dtype="bf16")
        s = Scatter(total_bytes=32, world=4, dim=1)
        g, p, preds, succs = _build_chain(r, s, n=4)
        optimize_comms(g)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), ReduceScatter)

    def test_gather_broadcast_to_allgather(self):
        ga = Gather(total_bytes=128, world=4, dim=0)
        b = Broadcast(total_bytes=128, world=4)
        g, p, preds, succs = _build_chain(ga, b, n=4)
        optimize_comms(g)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), AllGather)

    def test_gather_scatter_diff_dim_to_alltoall(self):
        ga = Gather(total_bytes=128, world=4, dim=0)
        s = Scatter(total_bytes=128, world=4, dim=1)
        g, p, preds, succs = _build_chain(ga, s, n=4)
        optimize_comms(g)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), AllToAll)

    def test_fused_collective_does_not_require_placement(self):
        gather = Gather(total_bytes=128, world=2, dim=0)
        broadcast = Broadcast(total_bytes=128, world=2)
        graph, placement, preds, succs = _build_chain(
            gather, broadcast, n=2)

        graph_only = ComputeGraph()
        for kernel in (*preds, gather, broadcast, *succs):
            graph_only.add_kernel(kernel)
        for index, pred in enumerate(preds):
            graph_only.add_data_edge(
                pred, gather, {"y": f"x{index}"})
        graph_only.add_data_edge(gather, broadcast, {"z": "z"})
        for index, succ in enumerate(succs):
            graph_only.add_data_edge(
                broadcast, succ, {f"y{index}": "a"})

        optimize_comms(graph_only)

        collective = next(
            kernel for kernel in graph_only.kernels
            if isinstance(kernel, AllGather)
        )
        assert len(collective.inputs) == 2
        assert len(collective.outputs) == 2
        assert all(
            placement.get_tensor_memory(tensor) is None
            for tensor in (*collective.inputs.values(),
                           *collective.outputs.values())
        )


class TestFuseWorld1Eliminated:
    def test_reduce_broadcast_world1_eliminated(self):
        r = Reduce(total_bytes=32, world=1, dtype="bf16")
        b = Broadcast(total_bytes=32, world=1)
        g, p, preds, succs = _build_chain(r, b, n=1)
        optimize_comms(g)
        assert g.kernels == frozenset(preds + succs)
        assert g._out_edges(preds[0])[0].dst is succs[0]

    def test_gather_scatter_same_dim_world1_eliminated(self):
        ga = Gather(total_bytes=32, world=1, dim=0)
        s = Scatter(total_bytes=32, world=1, dim=0)
        g, p, preds, succs = _build_chain(ga, s, n=1)
        optimize_comms(g)
        assert g.kernels == frozenset(preds + succs)

    def test_gather_broadcast_world1_eliminated(self):
        ga = Gather(total_bytes=32, world=1, dim=0)
        b = Broadcast(total_bytes=32, world=1)
        g, p, preds, succs = _build_chain(ga, b, n=1)
        optimize_comms(g)
        assert g.kernels == frozenset(preds + succs)


class TestBypass:
    def test_gather_scatter_same_dim_bypass(self):
        ga = Gather(total_bytes=128, world=4, dim=0)
        s = Scatter(total_bytes=128, world=4, dim=0)
        g, p, preds, succs = _build_chain(ga, s, n=1)
        optimize_comms(g)
        assert ga not in g.kernels
        assert s not in g.kernels
        assert g._out_edges(preds[0])[0].dst is succs[0]


class TestPlacementIndependentRewrite:
    @pytest.mark.parametrize("comm_cls", [Gather, Scatter])
    def test_same_direction_hierarchy_is_preserved(self, comm_cls):
        first = comm_cls(total_bytes=128, world=4, dim=0)
        second = comm_cls(total_bytes=128, world=2, dim=1)
        graph, placement, preds, succs = _build_chain(
            first, second, n=4, same_devices=False)

        optimize_comms(graph)

        assert first in graph.kernels
        assert second in graph.kernels
        assert set(preds + succs + [first, second]) == graph.kernels

    def test_gather_scatter_becomes_rank_direct_edges(self):
        gather = Gather(total_bytes=128, world=4, dim=0)
        scatter = Scatter(total_bytes=128, world=4, dim=0)
        g, p, preds, succs = _build_chain(
            gather, scatter, n=4, same_devices=False)

        optimize_comms(g)

        assert gather not in g.kernels
        assert scatter not in g.kernels
        for pred, succ in zip(preds, succs):
            assert g._out_edges(pred)[0].dst is succ

    def test_different_memories_do_not_affect_graph_rewrite(self):
        gather = Gather(total_bytes=SHARD.size_bytes, world=1, dim=0)
        scatter = Scatter(total_bytes=SHARD.size_bytes, world=1, dim=0)
        graph, placement, preds, succs = _build_chain(
            gather, scatter, n=1, same_devices=False)
        pred, succ = preds[0], succs[0]
        optimize_comms(graph)

        assert graph.kernels == frozenset({pred, succ})
        assert graph._out_edges(pred)[0].dst is succ

    def test_same_memory_uses_direct_edge(self):
        gather = Gather(total_bytes=128, world=1, dim=0)
        scatter = Scatter(total_bytes=128, world=1, dim=0)
        graph, placement, preds, succs = _build_chain(
            gather, scatter, n=1, same_devices=False)
        shared = Memory(name="shared", capacity_gb=1.0)
        placement.set_tensor_memory(preds[0].outputs["y"], shared)
        placement.set_tensor_memory(succs[0].inputs["a"], shared)

        optimize_comms(graph)

        assert graph.kernels == frozenset(preds + succs)
        assert graph._out_edges(preds[0])[0].dst is succs[0]

    def test_unrecognized_pair_is_preserved(self):
        first = AllGather(total_bytes=32, world=4)
        second = AllReduce(total_bytes=32, world=4, dtype="bf16")
        g, p, _, _ = _build_chain(
            first, second, n=4, same_devices=False)
        optimize_comms(g)
        assert first in g.kernels
        assert second in g.kernels

    def test_gather_scatter_different_dims_becomes_alltoall(self):
        gather = Gather(total_bytes=128, world=4, dim=0)
        scatter = Scatter(total_bytes=128, world=4, dim=1)
        g, p, _, _ = _build_chain(
            gather, scatter, n=4, same_devices=False)
        optimize_comms(g)
        assert any(isinstance(kernel, AllToAll) for kernel in g.kernels)
        assert gather not in g.kernels
        assert scatter not in g.kernels

    def test_gather_scatter_different_worlds_is_preserved(self):
        gather = Gather(total_bytes=128, world=4, dim=0)
        scatter = Scatter(total_bytes=128, world=2, dim=0)
        g, p, preds, succs = _build_chain(
            gather, scatter, n=4, same_devices=False)
        optimize_comms(g)
        assert gather in g.kernels
        assert scatter in g.kernels


class TestEliminateDead:
    @pytest.mark.parametrize("comm_cls,kwargs", [
        (Scatter, dict(total_bytes=128, world=4, dim=0)),
        (Broadcast, dict(total_bytes=32, world=4)),
        (Gather, dict(total_bytes=128, world=4, dim=0)),
        (Reduce, dict(total_bytes=32, world=4, dtype="bf16")),
        (AllReduce, dict(total_bytes=32, world=4, dtype="bf16")),
        (AllGather, dict(total_bytes=128, world=4)),
        (ReduceScatter, dict(total_bytes=32, world=4, dtype="bf16")),
        (AllToAll, dict(total_bytes=128, world=4)),
        (Send, dict(total_bytes=32)),
        (Recv, dict(total_bytes=32)),
    ])
    def test_single_edge_eliminated(self, comm_cls, kwargs):
        g = ComputeGraph()
        gpu = Compute(name="gpu0")
        pred = Kernel(outputs={"y": SHARD})
        succ = Kernel(inputs={"a": SHARD})
        comm = comm_cls(**kwargs)
        comm.inputs = {"x": SHARD}
        comm.outputs = {"y": SHARD}
        g.add_kernel(pred)
        g.add_kernel(comm)
        g.add_kernel(succ)
        g.add_data_edge(pred, comm, {"y": "x"})
        g.add_data_edge(comm, succ, {"y": "a"})
        p = Placement()
        p.set_kernel_device(pred, gpu)
        p.set_kernel_device(succ, gpu)
        optimize_comms(g)
        assert comm not in g.kernels
        assert g._out_edges(pred)[0].dst is succ

    def test_multi_edge_not_eliminated(self):
        g = ComputeGraph()
        gpu = Compute(name="gpu0")
        pred = Kernel(outputs={"y": SHARD})
        comm = Scatter(total_bytes=128, world=4, dim=0)
        comm.inputs = {"x": SHARD}
        comm.outputs = {"y0": SHARD, "y1": SHARD}
        succ0 = Kernel(inputs={"a": SHARD})
        succ1 = Kernel(inputs={"a": SHARD})
        g.add_kernel(pred)
        g.add_kernel(comm)
        g.add_kernel(succ0)
        g.add_kernel(succ1)
        g.add_data_edge(pred, comm, {"y": "x"})
        g.add_data_edge(comm, succ0, {"y0": "a"})
        g.add_data_edge(comm, succ1, {"y1": "a"})
        p = Placement()
        p.set_kernel_device(pred, gpu)
        p.set_kernel_device(succ0, gpu)
        p.set_kernel_device(succ1, gpu)
        optimize_comms(g)
        assert comm in g.kernels

    def test_removed_neighbor_can_make_an_earlier_comm_trivial(self):
        graph = ComputeGraph()
        pred = Kernel(outputs={"y": SHARD})
        first = Scatter(total_bytes=128, world=2, dim=0)
        first.inputs = {"x": SHARD}
        first.outputs = {"y0": SHARD, "y1": SHARD}
        second = Send(total_bytes=32)
        second.inputs = {"x": SHARD}
        second.outputs = {"y": SHARD}
        succ = Kernel(inputs={"a": SHARD, "b": SHARD})
        for kernel in (pred, first, second, succ):
            graph.add_kernel(kernel)
        graph.add_data_edge(pred, first, {"y": "x"})
        graph.add_data_edge(first, second, {"y0": "x"})
        graph.add_data_edge(first, succ, {"y1": "b"})
        graph.add_data_edge(second, succ, {"y": "a"})

        optimize_comms(graph)

        assert graph.kernels == frozenset({pred, succ})


# ── Guard clause tests (_fuse_pairs skips) ───────────────────────────


class TestFusePairsGuards:
    def test_create_collective_rejects_unexpected_pair(self):
        with pytest.raises(ValueError, match="Unexpected pair"):
            _create_collective(
                Broadcast(total_bytes=8, world=2),
                Gather(total_bytes=8, world=2, dim=0),
            )

    def test_same_dim_mismatched_shards_are_preserved(self):
        gather = Gather(total_bytes=128.0, world=2, dim=0)
        scatter = Scatter(total_bytes=128.0, world=2, dim=0)
        graph, _, _, _ = _build_chain(gather, scatter, n=2)
        scatter.outputs["y0"] = Tensor("bf16", (2, 8))

        optimize_comms(graph)

        assert gather in graph.kernels
        assert scatter in graph.kernels

    def test_bypass_tolerates_an_unconsumed_rank(self):
        gather = Gather(total_bytes=128.0, world=2, dim=0)
        scatter = Scatter(total_bytes=128.0, world=2, dim=0)
        graph, _, preds, succs = _build_chain(gather, scatter, n=2)
        graph.remove_kernel(succs[1])

        optimize_comms(graph)

        assert gather not in graph.kernels
        assert scatter not in graph.kernels
        assert graph._out_edges(preds[0])[0].dst is succs[0]
        assert not graph._out_edges(preds[1])

    def test_collector_multi_successor_is_preserved(self):
        gather = Gather(total_bytes=128.0, world=2, dim=0)
        scatter = Scatter(total_bytes=128.0, world=2, dim=0)
        graph, placement, preds, succs = _build_chain(
            gather, scatter, n=2, same_devices=False)
        gather.outputs["extra"] = SHARD
        extra = Kernel(inputs={"x": SHARD})
        graph.add_kernel(extra)
        graph.add_data_edge(gather, extra, {"extra": "x"})
        source_device = placement.get_kernel_device(preds[0]).device
        placement.set_kernel_device(extra, source_device)
        placement.set_tensor_memory(
            gather.outputs["extra"],
            placement.get_tensor_memory(extra.inputs["x"]))

        optimize_comms(graph)
        assert gather in graph.kernels
        assert scatter in graph.kernels
        assert set(preds + succs + [extra, gather, scatter]) == graph.kernels

    def test_collector_multi_out_edges_no_fuse(self):
        r = Reduce(total_bytes=128.0, world=2, dtype="bf16")
        b = Broadcast(total_bytes=128.0, world=2)
        g, p, preds, succs = _build_chain(r, b, n=2)
        r.outputs["z2"] = Tensor("bf16", SHARD.shape)
        extra = Kernel(inputs={"q": Tensor("bf16", SHARD.shape)})
        g.add_kernel(extra)
        g.add_data_edge(r, extra, {"z2": "q"})
        device = p.get_kernel_device(preds[0]).device
        p.set_kernel_device(extra, device)
        p.set_tensor_memory(
            r.outputs["z2"], p.get_tensor_memory(extra.inputs["q"]))
        optimize_comms(g)
        assert r in g.kernels
        assert b in g.kernels

    def test_succ_multi_in_edges_no_fuse(self):
        r = Reduce(total_bytes=128.0, world=2, dtype="bf16")
        b = Broadcast(total_bytes=128.0, world=2)
        g, p, preds, succs = _build_chain(r, b, n=2)
        b.inputs["q"] = Tensor("bf16", SHARD.shape)
        extra = Kernel(outputs={"q": Tensor("bf16", SHARD.shape)})
        g.add_kernel(extra)
        g.add_data_edge(extra, b, {"q": "q"})
        device = p.get_kernel_device(preds[0]).device
        p.set_kernel_device(extra, device)
        p.set_tensor_memory(
            b.inputs["q"], p.get_tensor_memory(extra.outputs["q"]))
        optimize_comms(g)
        assert r in g.kernels
        assert b in g.kernels

    def test_world_mismatch_no_fuse(self):
        r = Reduce(total_bytes=128.0, world=2, dtype="bf16")
        b = Broadcast(total_bytes=128.0, world=4)
        g, p, _, _ = _build_chain(r, b, n=2)
        optimize_comms(g)
        assert r in g.kernels
        assert b in g.kernels

    def test_incomplete_rank_ports_are_not_fused(self):
        r = Reduce(total_bytes=128.0, world=3, dtype="bf16")
        b = Broadcast(total_bytes=128.0, world=3)
        g, p, _, succs = _build_chain(r, b, n=3)
        g.remove_kernel(succs[2])
        del b.outputs["y2"]
        optimize_comms(g)
        assert r in g.kernels
        assert b in g.kernels


def test_optimize_comms_bypasses_rank_aligned_gather_scatter():
    gather = Gather(total_bytes=SHARD.size_bytes, world=4, dim=0)
    scatter = Scatter(total_bytes=SHARD.size_bytes, world=4, dim=0)
    graph, _, preds, succs = _build_chain(gather, scatter, 4)
    optimize_comms(graph)

    assert gather not in graph.kernels
    assert scatter not in graph.kernels
    for pred, succ in zip(preds, succs):
        assert any(edge.src is pred and edge.dst is succ
                   for edge in graph._out_edges(pred))


