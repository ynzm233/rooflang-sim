"""MAGI-2 geometry and accounting regressions from the pinned source model."""

from copy import deepcopy
import itertools
import math

import pytest

from rooflang.programs.models.magi2_preview.accounting import (
    cpu_text_accounting, memory_accounting,
)
from rooflang.programs.models.magi2_preview.config import (
    Magi2Config, load_model_spec, load_workload, model_architecture,
    turbo_windows, window_attention_pairs,
)


@pytest.mark.parametrize("workflow,lengths,padded,pairs,pos", [
    ("i2v", [61304, 59704], 121008, 7322748032, 2029),
    ("t2v", [59608, 58023], 117648, 6919782482, 2014),
])
def test_pinned_request_geometry(workflow, lengths, padded, pairs, pos):
    workload = load_workload()
    workload["workflow"] = workflow
    config = Magi2Config.from_dict(workload)
    shape = config.shape_summary()
    assert shape["positive_text_tokens"] == pos
    assert shape["preview_latent_shape"] == [1, 48, 32, 32, 56]
    assert shape["refiner_latent_shape"] == [1, 48, 63, 68, 120]
    assert shape["audio_latent_shape"] == [1, 250, 64]
    assert shape["preview_sequence_lengths"] == lengths
    assert shape["preview_padded_tokens"] == padded
    assert shape["preview_attention_pairs"] == pairs
    assert sum(shape["preview_modality_tokens"].values()) == padded
    assert shape["preview_forward_calls"] == 100
    assert shape["refiner_forward_calls"] == 10
    assert shape["decoded_video_shape"] == [249, 1088, 1920, 3]
    assert shape["resampled_audio_samples"] == 441000
    assert shape["turbo_windows"]["input_lengths"] == [8, 9, 9, 9, 9, 9, 9, 9, 8]
    assert config.dit_tokens == padded
    assert config.runtime_multimodal_config()["shape_summary"] == shape


@pytest.mark.parametrize("resolution,video,decoded", [
    ("272p", 14336, [125, 256, 448, 3]),
    ("540p", 57344, [125, 512, 896, 3]),
])
def test_lower_resolution_has_no_refiner_work(resolution, video, decoded):
    config = Magi2Config.from_dict({"resolution": resolution, "preview_steps": 3})
    shape = config.shape_summary()
    assert shape["preview_video_tokens"] == video
    assert shape["preview_forward_calls"] == 3
    assert shape["refiner_forward_calls"] == 0
    assert shape["refiner_latent_shape"] is None
    assert shape["decoded_video_shape"] == decoded
    assert "refiner_denoise" not in memory_accounting(shape)["stages"]


def test_only_selected_workflow_profile_is_resolved():
    workload = load_workload()
    workload["workflow"] = "t2v"
    workload["workflows"]["i2v"]["image_width"] = "unused-invalid-width"
    before = deepcopy(workload)
    config = Magi2Config.from_dict(workload)
    shape = config.shape_summary()
    assert shape["image_tokens"] == 0
    assert shape["image_latent_shape"] is None
    assert shape["positive_text_tokens"] == 2014
    assert "image_vae_encode" not in memory_accounting(shape)["stages"]
    assert workload == before


def test_steps_and_cp_shards_are_request_specific():
    config = Magi2Config.from_dict({
        "preview_steps": 7, "refiner_steps": 3, "cp_size": 4, "ep_size": 4,
        "positive_text_tokens": 31, "negative_text_tokens": 9,
    })
    shape = config.shape_summary()
    assert shape["preview_forward_calls"] == 7
    assert shape["refiner_forward_calls"] == 6
    assert shape["preview_per_rank_tokens"] * 4 == shape["preview_padded_tokens"]
    assert sum(shape["refiner_cond_per_rank_tokens"]) == 514080 + 31
    assert sum(shape["refiner_uncond_per_rank_tokens"]) == 514080 + 9


def test_window_attention_matches_independent_pair_enumeration():
    shape, blocks, radius, text = (3, 3, 2), (2, 2, 1), 1, 3
    points = list(itertools.product(*(range(size) for size in shape)))
    allowed_pairs = sum(
        all(abs(left[axis] // blocks[axis] - right[axis] // blocks[axis]) <= radius
            for axis in range(3))
        for left in points for right in points
    )
    expected = allowed_pairs + 2 * math.prod(shape) * text + text * text
    assert window_attention_pairs(shape, text, blocks, radius) == expected
    assert turbo_windows(63)["processed_latent_frames"] == 79


@pytest.mark.parametrize("overrides,match", [
    ({"preview_steps": 0}, "positive integer"),
    ({"cp_size": 4, "ep_size": 8}, "CP=EP"),
    ({"seconds": 5}, "10 seconds"),
    ({"cpu_text_timing_included": True}, "uncalibrated"),
])
def test_invalid_model_scenarios_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        Magi2Config.from_dict(overrides)


def test_cpu_text_preserves_work_and_unknown_timing():
    shape = Magi2Config.from_dict({}).shape_summary()
    result = cpu_text_accounting(shape)
    assert result["device"] == "cpu"
    assert result["forward_calls"] == 2
    assert result["parameter_count"] == 25624600064
    assert result["weight_bytes"] == 51249200128
    assert result["total_flops"] == 121614860632064
    assert result["execution_time_ms"] is None
    assert result["included_in_gpu_time"] is False
    assert [call["tokens"] for call in result["calls"]] == [2029, 429]


def test_raw_architecture_overrides_drive_geometry_and_accounting():
    raw = load_model_spec()
    shape = Magi2Config.from_dict({}).shape_summary()
    baseline = memory_accounting(shape, model_config=raw)
    updated = deepcopy(raw)
    updated["indexed_weight_bytes"]["preview_transformer"] += 4096
    updated["indexed_weight_bytes"]["refiner_transformer"] += 8192
    changed = memory_accounting(shape, model_config=updated)
    assert changed["all_resident_weight_bytes"] == baseline["all_resident_weight_bytes"] + 12288
    text = next(component for component in updated["components"] if component["name"] == "qwen_text")
    text["intermediate_size"] *= 2
    assert cpu_text_accounting(shape, updated)["total_flops"] > cpu_text_accounting(shape, raw)["total_flops"]
    preview = next(component for component in updated["components"] if component["name"] == "preview_transformer")
    preview["preview_latent_frames"] = 16
    config = Magi2Config.from_dict({})
    config.compute_mm_tokens(updated)
    assert config.shape_summary()["preview_video_tokens"] == 28672
    assert model_architecture()["preview_transformer"]["preview_latent_frames"] == 32


def test_architecture_layer_indices_validate_after_override():
    raw = load_model_spec()
    preview = next(component for component in raw["components"] if component["name"] == "preview_transformer")
    preview["num_layers"] = 1
    with pytest.raises(ValueError, match="modality_bank_layers"):
        model_architecture(raw)
