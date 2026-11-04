"""Nested request inputs must retain their declared dimensions and duration."""

import pytest

from rooflang.programs.models.minimax_h3.workflow import resolve_config


@pytest.mark.parametrize("field,value", [
    (field, value)
    for field, dimension in (("frames", 124), ("height", 768), ("width", 1344))
    for value in (dimension + 0.9, str(dimension), True)
])
def test_reference_geometry_rejects_coercion(field, value):
    reference = {"kind": "video", "frames": 124, "height": 768, "width": 1344}
    reference[field] = value
    with pytest.raises(ValueError, match=f"reference {field} must be an integer"):
        resolve_config({"workflow": "ref2va", "references": [reference]})


@pytest.mark.parametrize("duration", [True, "5", None, -1, float("nan"), float("inf")])
def test_visual_reference_rejects_invalid_audio_duration(duration):
    reference = {"kind": "image", "height": 768, "width": 1344,
                 "audio_seconds": duration}
    with pytest.raises(ValueError, match="audio_seconds must be finite and nonnegative"):
        resolve_config({"workflow": "ref2va", "references": [reference]})


@pytest.mark.parametrize("duration", [0, 5, 5.5])
def test_valid_reference_values_keep_workload_accounting(duration):
    reference = {"kind": "video", "frames": 124, "height": 768, "width": 1344,
                 "audio_seconds": duration}
    config, _, _ = resolve_config({"workflow": "ref2va", "references": [reference]})
    assert vars(config.references[0]) == reference
    assert config.shape_summary().packed_rows == 81118 + 2 * round(duration * 40)


def test_audio_only_reference_still_requires_positive_duration():
    references = [{"kind": "image", "height": 768, "width": 1344},
                  {"kind": "audio", "audio_seconds": 0}]
    with pytest.raises(ValueError, match="audio reference duration must be positive"):
        resolve_config({"workflow": "ref2va", "references": references})
