"""Independent arithmetic and routing checks for Ascend inference attention."""

from dataclasses import asdict

import pytest

from rooflang.programs.models.magi2_preview.operators import (
    FlashAttentionOperator,
    NpuFusedInferAttentionScoreOperator,
    NpuFusedInferAttentionScoreParams,
    QuantConfig,
)
from rooflang.programs.models.minimax_h3.bridge import build_component
from rooflang.programs.models.minimax_h3.hardware import A5PRServer
from rooflang.programs.presets.ascend910b import Ascend910B3


def packed_attention(**updates):
    arguments = {
        "num_heads": 4,
        "num_key_value_heads": 2,
        "q_len": 5,
        "kv_len": 5,
        "head_dim": 128,
        "input_layout": "TND",
        "actual_seq_lengths": [3, 5],
        "actual_seq_lengths_kv": [3, 5],
        "quant_config": QuantConfig(compute_dtype="bf16"),
        **updates,
    }
    return NpuFusedInferAttentionScoreOperator(**arguments)


def test_tnd_cumulative_boundaries_prevent_cross_document_attention():
    op = packed_attention()
    # The two documents have 3 and 2 tokens: 9 + 4 pairs, not 5 * 5.
    cube = 4 * 4 * 128 * 13
    vector = 15 * 4 * 13  # scale(1) + softmax(14), all FP32
    assert op.get_flops() == cube + vector
    compute = op.get_compute_component()
    assert compute.unit_type == "mixed"
    assert compute.cube_flops == cube
    assert compute.vector_flops == vector
    assert compute.dtype == "bf16"
    assert compute.get_unit_dtype("cube") == "bf16"
    assert compute.get_unit_dtype("vector") == "fp32"
    assert isinstance(op.params, NpuFusedInferAttentionScoreParams)
    assert asdict(op.params)["actual_seq_lengths"] == [3, 5]


def test_bnsd_reads_active_rows_but_writes_stored_output_rows():
    op = packed_attention(q_len=8, kv_len=9, input_layout="BNSD",
                          actual_seq_lengths=[5], actual_seq_lengths_kv=[5])
    # Q reads 5 rows, K/V each read 5, O writes all 8 stored rows.
    expected = (5 * 4 * 128 + 2 * 5 * 2 * 128 + 8 * 4 * 128) * 2 + 4
    details = op.get_memory_details()
    assert details["activation"] == expected
    assert details["kv_cache_read"] == details["kv_cache_saved"] == 0
    assert details["weight_read"] == details["weight_saved"] == 0
    assert op.get_memory_bytes() == expected
    assert op.get_flops() == 4 * 25 * (4 * 128 + 15)


def test_tnd_q_and_kv_documents_can_have_different_lengths():
    op = packed_attention(kv_len=6, actual_seq_lengths_kv=[2, 6])
    # Q documents [3, 2], KV documents [2, 4]: 3*2 + 2*4 = 14.
    assert op.get_flops() == 4 * 14 * (4 * 128 + 15)


def test_bnsd_causal_gqa_counts_triangle_and_mask_without_kv_cache():
    op = NpuFusedInferAttentionScoreOperator(
        num_heads=4, num_key_value_heads=2, q_len=4, kv_len=4,
        head_dim=128, input_layout="BNSD", causal=True,
        quant_config=QuantConfig(compute_dtype="bf16", kv_storage_dtype="fp32"),
    )
    # Each head has 1 + 2 + 3 + 4 valid pairs. The bool mask is shared.
    assert op.get_flops() == 4 * 10 * (4 * 128 + 15)
    assert op.get_memory_bytes() == (2 * 4 * 4 * 128 + 2 * 4 * 2 * 128) * 2 + 16 + 4
    details = op.get_memory_details()
    assert details["activation"] == op.get_memory_bytes()
    assert details["kv_cache_read"] == details["kv_cache_saved"] == 0


def test_bnsd_actual_lengths_are_per_batch_not_cumulative():
    op = NpuFusedInferAttentionScoreOperator(
        batch_size=2, num_heads=2, num_key_value_heads=1,
        q_len=5, kv_len=5, head_dim=128, input_layout="BNSD",
        actual_seq_lengths=[2, 3], actual_seq_lengths_kv=[4, 1],
    )
    # 2*4 + 3*1 = 11 pairs. Output retains 2*5 rows.
    assert op.get_flops() == 2 * 11 * (4 * 128 + 15)
    assert op.get_memory_bytes() == (5 * 2 * 128 + 2 * 5 * 128 + 10 * 2 * 128) * 2 + 4


def test_zero_kv_heads_selects_mha_and_default_bnsd_lengths_cover_storage():
    op = NpuFusedInferAttentionScoreOperator(
        batch_size=2, num_heads=4, q_len=3, kv_len=3,
        head_dim=128, input_layout="BNSD",
    )
    assert op.get_flops() == 2 * 4 * 9 * (4 * 128 + 15)
    assert op.get_memory_bytes() == 4 * 2 * 3 * 4 * 128 * 2 + 4


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_lse_output_uses_fp32_and_does_not_change_mathematical_flops(dtype):
    kwargs = {"quant_config": QuantConfig(compute_dtype=dtype)}
    plain = packed_attention(**kwargs)
    with_lse = packed_attention(**kwargs, softmax_lse_flag=True)
    assert with_lse.get_flops() == plain.get_flops()
    # A 5-row, 4-head FP32 LSE replaces the four-byte placeholder.
    assert with_lse.get_memory_bytes() - plain.get_memory_bytes() == 5 * 4 * 4 - 4


@pytest.mark.parametrize("updates", [
    {"actual_seq_lengths": None},
    {"actual_seq_lengths_kv": None},
    {"actual_seq_lengths": [3, 2]},
    {"actual_seq_lengths": [3, 6]},
    {"actual_seq_lengths": [3, 4]},
    {"actual_seq_lengths": [3, 5, 5]},
    {"actual_seq_lengths": [3.0, 5]},
    {"batch_size": 2},
    {"num_key_value_heads": 3},
    {"causal": True},
])
def test_invalid_packed_geometry_is_rejected(updates):
    with pytest.raises((TypeError, ValueError)):
        packed_attention(**updates)


@pytest.mark.parametrize("updates", [
    {"input_layout": "BSH"},
    {"sparse_mode": 1},
    {"inner_precise": 1},
    {"pre_tokens": 16},
    {"next_tokens": 0},
    {"scale": float("nan")},
    {"quant_config": QuantConfig(compute_dtype="fp32")},
    {"quant_config": QuantConfig(compute_dtype="fp8")},
])
def test_unmodeled_modes_are_rejected_instead_of_silently_approximated(updates):
    with pytest.raises((TypeError, ValueError)):
        packed_attention(**updates)


@pytest.mark.parametrize("updates", [
    {"actual_seq_lengths": [2, 3]},
    {"actual_seq_lengths": [5]},
    {"causal": True, "kv_len": 5},
])
def test_invalid_bnsd_lengths_and_causal_cross_attention_are_rejected(updates):
    with pytest.raises((TypeError, ValueError)):
        NpuFusedInferAttentionScoreOperator(
            **{"q_len": 4, "kv_len": 4, "input_layout": "BNSD", **updates},
        )


def attention_architecture(component):
    block = {
        "type": "attn-gqa" if component == "qwen_text" else "attn-mha",
        "num_heads": 4, "head_dim": 128,
        "causal": component == "qwen_text",
    }
    if component == "qwen_text":
        block["kv_heads"] = 2
    if component == "qwen_vision":
        block["type"] = "minimax-h3-qwen-vision-attn"
    return {"components": [{
        "name": component, "component_type": "dit", "hidden_size": 512,
        "layer_blocks": [{"layers": [{"blocks": [block]}]}],
    }]}


@pytest.mark.parametrize("component", [
    "qwen_vision", "qwen_text", "transformer_text_refiner",
    "transformer", "transformer_ref", "video_vae_decoder",
])
@pytest.mark.parametrize("sp", [1, 2])
def test_910b3_routes_every_h3_attention_component_and_preserves_dtype(component, sp):
    mm = {"execution_profile": "sglang_v0_5_21", "qwen_attention_segments": [3, 5]}
    graph, _ = build_component(component, attention_architecture(component), mm, 8,
                               Ascend910B3(sp), sp=sp)
    kernels = [k for k in graph.topological_sort() if hasattr(k, "operator")]
    attention = [k for k in kernels if isinstance(k.operator, NpuFusedInferAttentionScoreOperator)]
    assert attention
    assert not any(type(k.operator) is FlashAttentionOperator for k in kernels)
    dtype = "fp16" if component == "video_vae_decoder" else "bf16"
    for kernel in attention:
        assert kernel.operator.compute_dtype == dtype
        assert set(kernel.flops_by_dtype) == {dtype + ":cube", "fp32:vector"}
        assert kernel.csv_metadata["vector_compute_dtype"] == "fp32"
        assert kernel.csv_metadata["operator"] == "NpuFusedInferAttentionScoreOperator"
        assert kernel.csv_metadata["shape"] == asdict(kernel.operator.params)


def test_other_hardware_preserves_generic_flash_attention():
    graph, _ = build_component("transformer_ref", attention_architecture("transformer_ref"),
                               {}, 8, A5PRServer(1), sp=1)
    operators = [k.operator for k in graph.topological_sort() if hasattr(k, "operator")]
    assert any(type(op) is FlashAttentionOperator for op in operators)
    assert not any(isinstance(op, NpuFusedInferAttentionScoreOperator) for op in operators)


@pytest.mark.parametrize("component", ["transformer", "transformer_ref"])
@pytest.mark.parametrize("sp", [1, 2])
def test_910b3_dit_pads_to_64_and_attention_keeps_tail_separate(component, sp):
    seq_len = 82749
    padded = ((seq_len + 63) // 64) * 64
    graph, _ = build_component(component, attention_architecture(component),
                               {"execution_profile": "sglang_v0_5_21"}, seq_len,
                               Ascend910B3(sp), sp=sp)
    attention = [k.operator for k in graph.topological_sort()
                 if isinstance(getattr(k, "operator", None), NpuFusedInferAttentionScoreOperator)]
    assert len(attention) == 1
    op = attention[0]
    assert padded == 82752
    assert op.params.input_layout == "TND"
    assert op.params.q_len == op.params.kv_len == padded
    assert op.params.actual_seq_lengths == [seq_len, padded]
    assert op.params.actual_seq_lengths_kv == [seq_len, padded]
    assert op.get_flops() == (4 // sp) * (seq_len**2 + 3**2) * (4 * 128 + 15)


def test_910b3_text_refiner_stays_unsharded_when_main_sp_is_two():
    component = "transformer_text_refiner"
    graph, _ = build_component(component, attention_architecture(component),
                               {"execution_profile": "sglang_v0_5_21"}, 6915,
                               Ascend910B3(2), sp=2)
    kernels = graph.topological_sort()
    attention = [k.operator for k in kernels
                 if isinstance(getattr(k, "operator", None), NpuFusedInferAttentionScoreOperator)]
    assert len(attention) == 1
    op = attention[0]
    assert op.params.q_len == op.params.kv_len == 6915
    assert op.params.num_heads == 4
    assert op.params.actual_seq_lengths == [6915]
    assert not any(k.csv_metadata["operator"] == "AllToAllOperator" for k in kernels)
    assert not any("AllToAll" in k.csv_metadata["node_name"] for k in kernels)


def test_other_hardware_retains_legacy_refiner_sequence_parallelism():
    component = "transformer_text_refiner"
    graph, _ = build_component(component, attention_architecture(component), {}, 6915,
                               A5PRServer(2), sp=2)
    operators = [k.operator for k in graph.topological_sort() if hasattr(k, "operator")]
    op = next(op for op in operators if type(op) is FlashAttentionOperator)
    assert op.params.q_len == 6916
    assert op.params.num_heads == 2


@pytest.mark.parametrize("component", [
    "qwen_vision", "qwen_text", "transformer_text_refiner",
    "transformer", "transformer_ref", "video_vae_decoder",
])
@pytest.mark.parametrize("sp", [1, 2])
def test_910b3_explicit_generic_fa_preserves_original_operator_cost(component, sp):
    from rooflang.programs.models.minimax_h3.bridge import component_nodes

    mm = {"execution_profile": "sglang_v0_5_21", "qwen_attention_segments": [3, 5],
          "attention_backend": "flash_attention"}
    architecture = attention_architecture(component)
    graph, _ = build_component(component, architecture, mm, 9, Ascend910B3(sp), sp=sp)
    kernels = [k for k in graph.topological_sort() if hasattr(k, "operator")]
    actual = [k.operator for k in kernels if type(k.operator) is FlashAttentionOperator]
    expected = [n.operator for n in component_nodes(
        component, architecture, mm, 9, 1 if component == "transformer_text_refiner" else sp)
        if type(n.operator) is FlashAttentionOperator]
    assert actual and len(actual) == len(expected)
    assert not any(isinstance(k.operator, NpuFusedInferAttentionScoreOperator) for k in kernels)
    for op, original in zip(actual, expected):
        assert asdict(op.params) == asdict(original.params)
        assert op.get_flops() == original.get_flops()
        assert op.get_memory_details() == original.get_memory_details()
        assert op.compute_dtype == original.compute_dtype


def test_generic_fa_workflow_records_selection_and_rejects_unknown_backend():
    from rooflang.programs.models.minimax_h3.workflow import create_workflow, resolve_config

    workflow = create_workflow(Ascend910B3(2), {
        "workflow": "ref2va", "sp": 2, "execution_profile": "sglang_v0_5_21",
        "attention_backend": "flash_attention",
    })
    metadata = workflow.metadata
    assert metadata["attention_model"]["operator"] == "flash_attention"
    assert metadata["attention_model"]["text_refiner_sp"] == 1
    assert metadata["_resolved_config"]["workload"]["attention_backend"] == "flash_attention"
    with pytest.raises(ValueError, match="attention_backend"):
        resolve_config({"attention_backend": "flash_atention"})


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("length", [5, 4096])
def test_fia_precision_and_memory_efficiencies_reach_native_simulator(dtype, length):
    from rooflang.language.graph import ComputeGraph
    from rooflang.language.placement import Placement
    from rooflang.programs.models.magi2_preview.bridge import AnalyticalKernel
    from rooflang.programs.models.minimax_h3.bridge import H3Kernel, _set_memory_efficiency
    from rooflang.runtime.simulator import Simulator

    op = packed_attention(q_len=length, kv_len=length,
                          actual_seq_lengths=[length], actual_seq_lengths_kv=[length],
                          quant_config=QuantConfig(compute_dtype=dtype))
    hardware = Ascend910B3()
    kernel = H3Kernel(op, {}, "test")
    _set_memory_efficiency(kernel, hardware)
    graph = ComputeGraph()
    graph.add_kernel(kernel)
    placement = Placement(hardware)
    placement.set_kernel_device(kernel, hardware.devices[0])
    result = Simulator(graph, placement, hardware).run()
    # Q has 4 heads, K/V 2 each; O retains 4 heads and a 4-byte LSE placeholder.
    qkv_bytes = length * (4 + 2 * 2) * 128 * 2
    out_lse_bytes = length * 4 * 128 * 2 + 4
    compute_us = ((4 * 4 * 128 * length**2) / (294.912e6 * 0.7)
                  + (15 * 4 * length**2) / (9.216e6 * 0.7))
    memory_us = qkv_bytes / (1600e3 * 0.6) + out_lse_bytes / (1600e3 * 0.3)
    assert result.trace[0].compute_time_us == pytest.approx(compute_us)
    assert result.trace[0].memory_time_us == pytest.approx(memory_us)
    assert result.total_time_us == pytest.approx(max(compute_us, memory_us))
    assert kernel.transferred_bytes == qkv_bytes + out_lse_bytes
    assert op.get_memory_bytes_by_unit() == {"cube": qkv_bytes, "vector": out_lse_bytes}
    # The generic bridge also preserves mixed precision without unit suffixes.
    assert AnalyticalKernel(op, {}, "generic").flops_by_dtype == {
        dtype: 4 * 4 * 128 * length**2, "fp32": 15 * 4 * length**2,
    }


def test_b3_pure_units_receive_distinct_memory_efficiencies():
    graph, _ = build_component("transformer", attention_architecture("transformer"),
                               {"execution_profile": "sglang_v0_5_21"}, 8, Ascend910B3())
    seen = set()
    for kernel in graph.topological_sort():
        c = kernel.operator.get_compute_component()
        if c and c.unit_type in ("cube", "vector"):
            expected = 0.6 if c.unit_type == "cube" else 0.3
            assert kernel.local_memory_efficiency == pytest.approx(expected)
            assert kernel.transferred_bytes == kernel.operator.get_memory_bytes()
            seen.add(c.unit_type)
    assert seen == {"cube", "vector"}
