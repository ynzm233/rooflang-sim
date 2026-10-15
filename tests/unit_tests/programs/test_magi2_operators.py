"""Independent arithmetic checks for MAGI-2's analytical cost descriptions."""

import math

import pytest

from rooflang.programs.models.magi2_preview.operators import (
    Conv1dOperator,
    Conv3dOperator,
    ConvTranspose1dOperator,
    FlashAttentionOperator,
    FlashAttentionSinkOperator,
    FlashMHMoEOperator,
    MoERoutingOperator,
    QuantConfig,
    RmsNormOperator,
    TensorMoveOperator,
)


def test_attention_causal_prefix_pairs_and_kv_precision():
    op = FlashAttentionOperator(
        batch_size=2, num_heads=8, kv_heads=2, q_len=3, kv_len=7,
        head_dim=16, causal=True,
        quant_config=QuantConfig(compute_dtype="bf16", kv_storage_dtype="fp32"),
    )
    # The three queries see 5, 6 and 7 keys after their four-token prefix.
    assert op.get_flops() == 2 * 8 * (5 + 6 + 7) * (4 * 16 + 14)
    assert op.get_memory_details()["activation"] == 2 * 8 * 3 * 16 * 2 * 2
    assert op.get_memory_details()["kv_cache_read"] == 2 * 2 * 7 * 16 * 2 * 4


def test_attention_sink_retains_mixed_precision_sequential_subcalls():
    op = FlashAttentionSinkOperator(
        batch_size=1, num_heads=4, kv_heads=2, q_len=8, kv_len=8,
        head_dim=16, attention_pairs=40, sink_weight_heads=24,
        quant_config=QuantConfig(compute_dtype="bf16"),
    )
    stages = op.get_stages()
    names = [stage.name for stage in stages]
    assert names[0] == "flash_attention"
    assert names.index("correct_lse") < names.index("scale_output")
    assert names.index("scale_output") < names.index("copy_corrected_lse")
    attention = stages[0].operator().get_compute_component()
    assert attention.unit_type == "mixed"
    assert attention.cube_flops == 4 * 40 * 4 * 16
    assert attention.vector_flops == 14 * 40 * 4
    assert attention.dtype == "bf16"
    assert all(stage.compute_dtype == "fp32" for stage in stages[1:])
    assert op.get_memory_details()["weight_saved"] == 24 * 4
    assert op.get_flops() == sum(stage.operator().get_flops() for stage in stages)
    assert op.get_memory_bytes() == sum(stage.operator().get_memory_bytes() for stage in stages)
    # The final transpose is a real memory copy despite having no arithmetic.
    copy = stages[-1].operator()
    assert copy.get_flops() == 0
    assert copy.get_memory_bytes() == 8 * 4 * 4 * 2
    assert copy.get_compute_component().unit_type == "vector"


def test_attention_sensitivity_preserves_mathematical_flops():
    args = dict(num_heads=4, kv_heads=2, q_len=8, kv_len=8, head_dim=16)
    base = FlashAttentionSinkOperator(**args)
    amplified = FlashAttentionSinkOperator(**args, softmax_compute_multiplier=3)
    assert amplified.get_flops() == base.get_flops()
    base_compute = base.get_stages()[0].operator().get_compute_component()
    amplified_compute = amplified.get_stages()[0].operator().get_compute_component()
    assert amplified_compute.cube_flops == base_compute.cube_flops
    assert amplified_compute.vector_flops == 3 * base_compute.vector_flops


def test_moe_active_experts_change_traffic_but_not_residency_or_flops():
    args = dict(tokens=5, num_heads=2, num_experts=8, top_k=2,
                head_dim=4, intermediate_size=7)
    balanced = FlashMHMoEOperator(**args)
    concentrated = FlashMHMoEOperator(**args, active_experts_per_head=2)
    assert balanced.get_cube_flops() == 6 * 5 * 2 * 2 * 4 * 7
    assert balanced.get_vector_flops() == 20 * 7 * 17 + 2 * 20 * 4
    assert balanced.get_flops() == concentrated.get_flops()
    assert balanced.get_memory_details()["weight_saved"] == 2 * 8 * 3 * 4 * 7 * 2
    assert balanced.get_memory_details()["weight_saved"] == concentrated.get_memory_details()["weight_saved"]
    assert balanced.get_memory_details()["weight_read"] == 4 * concentrated.get_memory_details()["weight_read"]
    # BF16 boundaries, int32 IDs, FP32 probabilities and int64 expert offsets.
    assert balanced.get_memory_details()["activation"] == 2 * 5 * 2 * 4 * 2 + 20 * 8 + 17 * 8


def test_moe_compute_amplification_is_separate_from_reported_work():
    args = dict(tokens=5, num_heads=2, num_experts=8, top_k=2,
                head_dim=4, intermediate_size=7)
    base = FlashMHMoEOperator(**args)
    amplified = FlashMHMoEOperator(**args, compute_amplification=2.5)
    assert amplified.get_flops() == base.get_flops()
    assert amplified.get_compute_component().cube_flops == 2.5 * base.get_cube_flops()
    assert amplified.get_compute_component().vector_flops == base.get_vector_flops()


def test_router_subcalls_preserve_sort_order_and_index_widths():
    op = MoERoutingOperator(tokens=5, num_heads=2, num_experts=8, top_k=3)
    stages = {stage.name: stage for stage in op.get_stages()}
    names = list(stages)
    assert names.index("topk") < names.index("stable_argsort")
    assert names.index("stable_argsort") < names.index("cast_gather_ids_int32")
    assert stages["scores_sigmoid"].vector_flops == 5 * 2 * 8 * 10
    assert stages["topk"].vector_flops == pytest.approx(5 * 2 * 8 * math.log2(3))
    assert stages["topk"].activation_bytes == 5 * 2 * 8 * 4 + 5 * 2 * 3 * (4 + 8)
    assert stages["cast_gather_ids_int32"].activation_bytes == 5 * 2 * 3 * (8 + 4)
    assert stages["stable_argsort"].vector_flops == 30 * math.ceil(math.log2(30))
    assert op.get_memory_details()["weight_saved"] == 2 * 8 * 4


def test_dilated_grouped_conv1d_counts_taps_not_receptive_field():
    op = Conv1dOperator(
        batch=2, c_in=4, c_out=6, L_in=10, kernel_size=3, dilation=2,
        pad=2, group=2, has_bias=True, cal_kv_cache=True,
    )
    assert op.L_out == 10
    assert op.get_flops() == 2 * 2 * 2 * 6 * 3 * 10 + 2 * 6 * 10
    assert op.get_memory_details()["weight_saved"] == (2 * 6 * 3 + 6) * 2
    assert op.get_memory_details()["kv_cache_read"] == 2 * 4 * 4 * 2


def test_conv3d_bias_changes_weights_and_compute_not_activation_traffic():
    args = dict(batch=2, c_in=4, c_out=6, d_in=3, h_in=5, w_in=5,
                kernel_t=3, kernel_h=3, kernel_w=3, pad_t=1, pad_h=1, pad_w=1)
    plain = Conv3dOperator(**args)
    biased = Conv3dOperator(**args, has_bias=True)
    assert (plain.d_out, plain.h_out, plain.w_out) == (3, 5, 5)
    assert plain.get_flops() == 2 * 2 * 4 * 6 * 27 * 3 * 5 * 5
    assert biased.get_flops() - plain.get_flops() == 2 * 6 * 3 * 5 * 5
    assert biased.get_memory_details()["weight_saved"] - plain.get_memory_details()["weight_saved"] == 6 * 2
    assert biased.get_memory_details()["activation"] == plain.get_memory_details()["activation"]


def test_transpose_conv_output_padding_changes_geometry_not_tap_work():
    args = dict(c_in=2, c_out=4, L_in=3, kernel_size=3, stride=2, dilation=2, pad=1)
    plain = ConvTranspose1dOperator(**args)
    padded = ConvTranspose1dOperator(**args, output_padding=1)
    assert (plain.L_out, padded.L_out) == (7, 8)
    assert plain.get_flops() == padded.get_flops() == 2 * 3 * 2 * 4 * 3
    assert padded.get_memory_details()["activation"] - plain.get_memory_details()["activation"] == 4 * 2


@pytest.mark.parametrize("arguments", [
    {"group": 2, "c_in": 3}, {"stride": 2, "output_padding": 2},
    {"pad": 4, "L_in": 1}, {"dilation": 0},
])
def test_transpose_conv_rejects_impossible_geometry(arguments):
    with pytest.raises(ValueError):
        ConvTranspose1dOperator(**arguments)


def test_tensor_materialization_preserves_boundary_bytes_and_views_are_free():
    copied = TensorMoveOperator(mode="pad", input_elements=6, output_elements=10,
                                input_dtype="bf16", output_dtype="fp32")
    assert copied.get_flops() == 0
    assert copied.get_memory_bytes() == 6 * 2 + 10 * 4
    view = TensorMoveOperator(mode="reshape", input_elements=12,
                              output_elements=12, materialized=False)
    assert view.get_flops() == 0
    assert view.get_memory_bytes() == 0


def test_shared_norm_weights_retain_reads_without_duplicate_residency():
    op = RmsNormOperator(
        batch_seq=2, dim=8, weight_banks=3, shared_weight=True,
        input_dtype="bf16", output_dtype="bf16",
        quant_config=QuantConfig(compute_dtype="fp32", weight_storage_dtype="fp32"),
    )
    assert op.get_memory_details()["activation"] == 2 * 8 * 2 * 2
    assert op.get_memory_details()["weight_read"] == 3 * 8 * 4
    assert op.get_memory_details()["weight_saved"] == 0
