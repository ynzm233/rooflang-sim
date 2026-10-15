"""MAGI recipes execute as native scheduled work, including collectives."""
import pytest

from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.comm import CommKernel
from rooflang.programs.models.magi2_preview.bridge import (
    AnalyticalKernel, ModuleType, Node, _lower, build_component,
)
from rooflang.programs.models.magi2_preview.config import Magi2Config
from rooflang.programs.models.magi2_preview.operators import (
    FlashAttentionSinkOperator, OpType, QuantConfig,
)
from rooflang.programs.presets.b300 import B300Cluster
from rooflang.programs.presets.h200 import H200Cluster
from rooflang.runtime.simulator import Simulator


def _request(**overrides):
    return Magi2Config.from_dict({
        "resolution": "272p", "cp_size": 8, "ep_size": 8,
        "preview_steps": 1, "refiner_steps": 1, **overrides,
    })


@pytest.mark.parametrize("component,distributed,tag_mask,kv_dtypes", [
    ("image_vae_encoder", False, 0, {"fp32"}),
    ("preview_transformer", True, 256, {"bf16", "fp32", "int32"}),
    ("video_vae_decoder", False, 1024, {"bf16", "fp32"}),
    ("audio_vae_decoder", False, 1024, {"fp32"}),
])
def test_parameterized_recipes_execute_native_kernels(component, distributed, tag_mask, kv_dtypes):
    config = _request()
    hardware = B300Cluster()
    graph, placement = build_component(
        component, config.runtime_multimodal_config(), config.architecture,
        hardware, cp=8 if distributed else 1, ep=8 if distributed else 1,
    )
    result = Simulator(graph, placement, hardware).run()
    assert result.total_time_us > 0
    assert all(isinstance(kernel, (AnalyticalKernel, CommKernel)) for kernel in graph.kernels)
    assert {kernel.csv_metadata["tag_mask"] for kernel in graph.kernels} == {tag_mask}
    assert {kernel.csv_metadata["kv_storage_dtype"] for kernel in graph.kernels} == kv_dtypes
    for entry in result.trace:
        kernel = entry.kernel
        assert kernel.csv_metadata["component"] == component
        if isinstance(kernel, AnalyticalKernel):
            assert kernel.transferred_bytes == pytest.approx(kernel.operator.get_memory_bytes())
            assert entry.memory_time_us == pytest.approx(kernel.transferred_bytes / 7750e3)
    if distributed:
        assert len(graph.magi2_adapter_selection["adapter_input_rank_costs_us"]) == 8


def test_broadcast_uses_actual_mesh_and_hardware_links():
    config = _request()
    runtimes = []
    for hardware in (B300Cluster(), H200Cluster()):
        graph, placement = build_component(
            "condition_broadcast", {**config.runtime_multimodal_config(), "broadcast_kind": "text"},
            config.architecture, hardware, cp=8, ep=8,
        )
        result = Simulator(graph, placement, hardware).run()
        runtimes.append(result.total_time_us)
        assert len(graph.kernels) == 6 * 3  # six payloads, three binary-tree hops
        assert all(len(placement.infer_comm_devices(kernel)) == 8 for kernel in graph.kernels)
        assert all(entry.network_time_us > 0 for entry in result.trace)
        assert {kernel.csv_metadata["tag_mask"] for kernel in graph.kernels} == {0}
        assert {kernel.csv_metadata["kv_storage_dtype"] for kernel in graph.kernels} == {"int32"}
    assert runtimes[1] > runtimes[0]


def test_attention_composite_subcalls_keep_their_order_and_precision():
    op = FlashAttentionSinkOperator(
        batch_size=1, num_heads=2, kv_heads=2, q_len=32, kv_len=32,
        head_dim=16, sink_weight_heads=2,
        quant_config=QuantConfig(compute_dtype="bf16", weight_storage_dtype="bf16",
                                 kv_storage_dtype="bf16"),
    )
    node = Node(node_name="attention", operator=op, op_type=OpType.COMPUTE,
                module=ModuleType.ATTENTION, layer_idx=0, block_id="layer",
                shape="test", semantic="attention", assumption="", source_compute_dtype="bf16")
    hardware = B300Cluster()
    devices = sorted((node for node in hardware.nodes
                      if isinstance(node, Compute) and node.kind == "gpu"), key=lambda node: node.name)
    graph, placement = _lower([node], "attention_test", hardware, devices)
    result = Simulator(graph, placement, hardware).run()
    stages = op.get_stages()
    assert [entry.kernel.csv_metadata["subcall"] for entry in result.trace] == [stage.name for stage in stages]
    assert sum(entry.kernel.flops for entry in result.trace) == op.get_flops()
    assert {dtype for entry in result.trace for dtype in entry.kernel.flops_by_dtype} == {"bf16", "fp32"}
    assert {entry.kernel.csv_metadata["kv_storage_dtype"] for entry in result.trace} == {"bf16"}
    for previous, current in zip(result.trace, result.trace[1:]):
        assert current.start_us == pytest.approx(previous.end_us)


def test_geometry_and_architecture_change_native_work():
    hardware = B300Cluster()
    totals = []
    for resolution in ("272p", "540p"):
        config = _request(resolution=resolution)
        graph, _ = build_component(
            "image_vae_encoder", config.runtime_multimodal_config(), config.architecture,
            hardware,
        )
        totals.append(sum(kernel.flops for kernel in graph.kernels))
    assert totals[1] > totals[0]


def test_reject_mesh_larger_than_actual_hardware():
    config = _request()
    with pytest.raises(ValueError, match="available GPUs"):
        build_component("preview_transformer", config.runtime_multimodal_config(),
                        config.architecture, B300Cluster(), cp=16, ep=16)


def test_reject_asymmetric_hbm_for_aggregate_traffic_formulas():
    config = _request()
    hardware = B300Cluster()
    device = sorted((node for node in hardware.nodes
                     if isinstance(node, Compute) and node.kind == "gpu"), key=lambda node: node.name)[0]
    fabric = hardware.find_fabric(device, hardware.find_local_memory(device))
    fabric.src_to_dst_bandwidth_gbs *= 0.5
    with pytest.raises(ValueError, match="equal HBM read/write"):
        build_component("image_vae_encoder", config.runtime_multimodal_config(),
                        config.architecture, hardware)


def test_traffic_volumes_do_not_create_false_oom_for_single_rank_refiner():
    config = _request(resolution="1080p", cp_size=1, ep_size=1)
    hardware = H200Cluster()
    graph, placement = build_component(
        "refiner_transformer", config.runtime_multimodal_config(),
        config.architecture, hardware,
    )
    result = Simulator(graph, placement, hardware).run()
    assert result.total_time_us > 0
    assert max(result.peak_memory.values()) == 13482481664  # graph resident weights
    assert max(kernel.csv_metadata["activation_peak_bytes_proxy"]
               for kernel in graph.kernels) > 0
    assert all(kernel.outputs["traffic"].storage_bytes == 0
               for kernel in graph.kernels if isinstance(kernel, AnalyticalKernel))


@pytest.mark.parametrize("component,updates,distributed,expected", [
    ("condition_broadcast", {"broadcast_kind": "text"}, True,
     (0, 50339936, 0)),
    ("image_vae_encoder", {}, False,
     (2005969459200, 10036611456, 277900800)),
    ("preview_transformer", {}, True,
     (669829128179561, 737590200808, 1486946304)),
    ("refiner_transformer", {"branch": "cond"}, True,
     (1206261125164565, 1984703281996, 8453914560)),
    ("audio_vae_decoder", {}, False,
     (1266444073346, 19438884360, 786432000)),
])
def test_official_1080p_recipe_work_matches_reference_csv(
    component, updates, distributed, expected,
):
    # Independent regression totals from Markov's official 1080p I2V output.
    # These constrain workload formulas, not hardware-specific execution times;
    # no external repository or result file is needed when running this test.
    config = _request(resolution="1080p")
    graph, _ = build_component(
        component, {**config.runtime_multimodal_config(), **updates},
        config.architecture, B300Cluster(),
        cp=8 if distributed else 1, ep=8 if distributed else 1,
    )
    actual = (
        sum(kernel.flops for kernel in graph.kernels),
        sum(kernel.input_bytes + kernel.output_bytes + kernel.loaded_weight_bytes
            for kernel in graph.kernels),
        max(kernel.csv_metadata["activation_peak_bytes_proxy"] for kernel in graph.kernels),
    )
    assert actual == pytest.approx(expected, rel=1e-10)
