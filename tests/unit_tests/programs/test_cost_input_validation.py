"""Reject impossible analytical costs before deriving FLOPs or tensor traffic."""

import pytest

from rooflang.language.utils import dtype_bytes
from rooflang.programs.models.magi2_preview.operators import (
    Conv1dOperator, Conv3dOperator, FlashMHMoEOperator, GroupMatmulOperator,
    MatmulOperator, QuantConfig, RmsNormOperator, TensorMoveOperator,
)
from rooflang.programs.models.magi2_preview._cost.core import get_dtype_size
from rooflang.programs.models.magi2_preview._cost.topk import TopkOperator


@pytest.mark.parametrize("batch_seq", [-1, 0, 1.5, 2.0, True, "2", None])
def test_common_constructor_rejects_invalid_geometry(batch_seq):
    with pytest.raises((TypeError, ValueError), match="batch_seq"):
        MatmulOperator(batch_seq=batch_seq, in_features=2, out_features=3)


@pytest.mark.parametrize("resident_heads", [True, 1.5, "2"])
def test_optional_integer_geometry_is_not_coerced(resident_heads):
    with pytest.raises(TypeError, match="resident_heads"):
        FlashMHMoEOperator(resident_heads=resident_heads)


@pytest.mark.parametrize("operator, arguments", [
    (Conv1dOperator, {"stride": 0}),
    (Conv1dOperator, {"c_in": 3, "c_out": 4, "group": 2}),
    (Conv1dOperator, {"c_in": 4, "c_out": 3, "group": 2}),
    (Conv1dOperator, {"L_in": 1, "kernel_size": 3}),
    (Conv3dOperator, {"stride_t": 0}),
    (Conv3dOperator, {"stride_h": 0}),
    (Conv3dOperator, {"stride_w": 0}),
    (Conv3dOperator, {"c_in": 3, "c_out": 4, "groups": 2}),
    (Conv3dOperator, {"c_in": 4, "c_out": 3, "groups": 2}),
    (Conv3dOperator, {"d_in": 1, "kernel_t": 3}),
    (Conv3dOperator, {"kernel_t": 1, "h_in": 1, "kernel_h": 3}),
    (Conv3dOperator, {"kernel_t": 1, "w_in": 1, "kernel_w": 3}),
])
def test_convolution_rejects_invalid_geometry_before_division(operator, arguments):
    with pytest.raises(ValueError):
        operator(**arguments)


@pytest.mark.parametrize("tokens", [[], [-1, 2], [1.5], [True], ["2"], (1, 2)])
def test_group_matmul_requires_nonempty_integer_expert_counts(tokens):
    with pytest.raises((TypeError, ValueError), match="expert_token_list"):
        GroupMatmulOperator(expert_token_list=tokens, in_features=4, out_features=8)


def test_group_matmul_preserves_inactive_expert_residency():
    inactive = GroupMatmulOperator(expert_token_list=[0, 0], in_features=4, out_features=8)
    active = GroupMatmulOperator(expert_token_list=[0, 2], in_features=4, out_features=8)
    assert inactive.get_flops() == inactive.get_memory_bytes() == 0
    assert inactive.get_memory_details()["weight_saved"] == 128
    assert active.get_flops() == 128
    assert active.get_memory_details()["weight_saved"] == 128
    assert active.get_memory_details()["weight_read"] == 64
    assert active.get_memory_details()["activation"] == 48


@pytest.mark.parametrize("field", ["compute_dtype", "weight_storage_dtype", "kv_storage_dtype", "communicate_dtype"])
def test_quant_config_rejects_unknown_precision(field):
    with pytest.raises(ValueError, match="unknown dtype"):
        QuantConfig(**{field: "fp64"})


def test_boundary_dtype_cannot_silently_fall_back_to_fp16():
    with pytest.raises(ValueError, match="unknown dtype"):
        RmsNormOperator(batch_seq=2, dim=4, input_dtype="fp64")


@pytest.mark.parametrize("dtype", ["fp4", "fp8", "ue8m0", "bf16", "fp16", "fp32", "int32", "int64"])
def test_shared_dtype_widths_match_native_tensor_widths(dtype):
    assert get_dtype_size(dtype) == dtype_bytes(dtype)


def test_int64_cast_uses_eight_byte_boundary_and_subbyte_types_remain_valid():
    cast = TensorMoveOperator(mode="cast", input_elements=3, output_elements=3,
                              input_dtype="int32", output_dtype="int64")
    assert cast.get_memory_bytes() == 36
    assert get_dtype_size("bf4") == 0.5
    assert get_dtype_size("mix2.5") == 0.3125


def test_topk_includes_values_and_int64_indices_without_changing_flop_proxy():
    op = TopkOperator(batch_seq=2, dim=4, k=2)
    assert op.get_memory_bytes() == 56
    assert op.get_flops() == 8
    with pytest.raises(ValueError, match="k cannot exceed dim"):
        TopkOperator(dim=4, k=5)
