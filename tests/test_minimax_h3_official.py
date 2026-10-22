"""Execution-profile and capacity accounting checks independent of DMA timing."""
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from rooflang.language.kernels.kernel import Kernel
from rooflang.programs.models.minimax_h3 import workflow as workflow_module
from rooflang.programs.models.minimax_h3.hardware import A5PRServer
from rooflang.runtime.export_config import csv_fields, project_plan_metadata


def profile_config(**updates):
    return {
        "execution_profile": "sglang_v0_5_21",
        "workflow": "ref2va",
        "sp": 1,
        "num_inference_steps": 49,
        "prompt": "An exact official request prompt.",
        "text_condition_rows": 6915,
        "fps": 24,
        "references": [
            {"kind": "video", "frames": 124, "height": 768, "width": 1344,
             "audio_seconds": 124 / 24},
            {"kind": "audio", "audio_seconds": 124 / 24},
        ],
        "request_source": {"source_commit": "request-revision", "prompt_sha256": "test"},
        "layerwise_offload": {"prefetch_size": 1, "resident_layers": 0},
        **updates,
    }


def profile_hardware():
    hardware = A5PRServer(1)
    for device in (n for n in hardware.nodes if n.kind == "gpu"):
        for dtype in ("bf16", "fp32"):
            for unit in ("cube", "vector"):
                device.tflops[f"{dtype}:{unit}"] = device.tflops[f"fp16:{unit}"]
    return hardware


@pytest.fixture
def modeled_graphs(monkeypatch):
    """Isolate workflow accounting from the separately tested native DMA graph."""
    fixed = {
        "qwen_vision": 10, "qwen_text": 20, "video_vae_encoder": 30,
        "audio_vae_encoder": 40, "transformer_text_refiner": 50,
        "transformer_ref": 60, "video_vae_decoder": 70, "audio_vae_decoder": 80,
    }
    streamed = {name: (0 if name in ("video_vae_encoder", "audio_vae_encoder",
                                    "audio_vae_decoder") else 100)
                for name in fixed}
    calls = []

    def build(component, architecture, mm, seq, hardware, sp):
        calls.append((component, deepcopy(architecture), deepcopy(mm), sp))
        return component, None

    def simulate(component, placement, hardware):
        kernel = Kernel()
        kernel.csv_metadata = {
            "source_weight_saved": fixed[component] + streamed[component],
            "activation_peak_bytes_proxy": 7,
        }
        return SimpleNamespace(
            component=component, trace=[SimpleNamespace(kernel=kernel)],
            total_time_us=10_000)

    def stats(result, hardware):
        component = result.component
        amount = streamed[component]
        return {
            "fixed_weight_bytes": fixed[component],
            "streamed_weight_bytes": amount,
            "weight_peak_bytes": fixed[component] + 2 * amount,
            "host_to_device_bytes": amount,
            "copy_time_ms": 2 if amount else 0,
            "exposed_transfer_time_ms": 1 if amount else 0,
            "offloaded_layer_count": 1 if amount else 0,
        }

    monkeypatch.setattr(workflow_module, "build_component", build)
    monkeypatch.setattr(workflow_module, "simulate", simulate)
    monkeypatch.setitem(sys.modules, "rooflang.programs.models.minimax_h3.offload",
                        SimpleNamespace(graph_stats=stats))
    return calls, fixed


def test_profile_aligns_layers_refiner_calls_shapes_and_provenance(modeled_graphs):
    config = profile_config()
    original = deepcopy(config)
    workflow = workflow_module.create_workflow(profile_hardware(), config)
    results = workflow.run()
    assert config == original
    assert workflow.metadata["shape_summary"]["packed_rows"] == 82749
    assert workflow.metadata["operator_graph_multipliers"]["transformer_text_refiner"] == 1
    assert workflow.metadata["operator_graph_multipliers"]["transformer_denoise"] == 49
    assert workflow.metadata["refiner_forward_count"] == 1
    assert workflow.metadata["compute_dtype"] == "bf16"
    assert workflow.metadata["request_source"] == config["request_source"]
    assert workflow.metadata["stage_precision"]["condition_vae_encode"] == "fp32"
    assert results["transformer_denoise"].metrics["host_to_device_bytes"] == 4900
    assert results["transformer_text_refiner"].metrics["host_to_device_bytes"] == 100
    calls, _ = modeled_graphs
    architecture = calls[0][1]
    qwen = next(c for c in architecture["components"] if c["name"] == "qwen_text")
    assert qwen["layer_blocks"][0]["repeat_times"] == 50
    assert calls[0][2]["layerwise_offload"] == config["layerwise_offload"]
    legacy = workflow_module.create_workflow(A5PRServer(), {})
    legacy_qwen = next(c for c in legacy.metadata["_resolved_config"]["architecture"]["components"]
                       if c["name"] == "qwen_text")
    assert legacy_qwen["layer_blocks"][0]["repeat_times"] == 64


def test_global_fixed_weights_and_two_layer_peak_are_not_multiplied(modeled_graphs):
    workflow = workflow_module.create_workflow(profile_hardware(), profile_config())
    results = workflow.run()
    _, fixed = modeled_graphs
    fixed_total = sum(fixed.values())
    persistent = workflow.metadata["persistent_tensor_total_bytes"]
    assert workflow.metadata["fixed_weight_bytes"] == fixed_total
    # The two reference audio invocations share encoder weights.
    assert results["condition_vae_encode"].metrics["fixed_weight_bytes"] == 70
    assert workflow.metadata["peak_stage_memory_bytes"] == fixed_total + 200 + 7 + persistent
    assert workflow.metadata["memory_validity_basis"] == "native_layerwise_weights_plus_activation_proxy"
    assert workflow.metadata["all_resident_memory_bytes"] is None
    assert workflow.metadata["is_valid"]
    workflow_short = workflow_module.create_workflow(
        profile_hardware(), profile_config(num_inference_steps=1))
    workflow_short.run()
    # Timestep-index storage depends on step count; the native weight peak does not.
    short = workflow_short.metadata
    assert (workflow.metadata["peak_stage_memory_bytes"] - persistent
            == short["peak_stage_memory_bytes"] - short["persistent_tensor_total_bytes"])


def test_selected_stage_capacity_failure_preserves_transfer_metadata(modeled_graphs):
    hardware = profile_hardware()
    workflow = workflow_module.create_workflow(hardware, profile_config())
    workflow.run("transformer_denoise")
    boundary = workflow.metadata["peak_stage_memory_bytes"]
    for memory in (n for n in hardware.nodes if n.kind == "hbm"):
        memory.capacity_gb = (boundary - 1) / 1e9
    workflow.run("transformer_denoise")
    assert not workflow.metadata["is_valid"]
    assert workflow.metadata["memory_scope"] == "selected_stages_only"
    assert workflow.metadata["request_count"] is None
    assert workflow.metadata["packed_rows_per_second"] is None
    assert workflow.metadata["host_to_device_bytes"] == 4900
    assert workflow.metadata["refiner_forward_count"] == 0
    plan = project_plan_metadata("minimax_h3", "test", workflow.metadata)
    assert "host_to_device_bytes" in csv_fields("plan", "minimax_h3", [plan])
    assert plan["execution_profile"] == "sglang_v0_5_21"


@pytest.mark.parametrize("changes", [
    {"execution_profile": "unknown"},
    {"compute_dtype": "fp16"},
    {"request_source": []},
    {"layerwise_offload": {"prefetch_size": True, "resident_layers": 0}},
    {"layerwise_offload": {"prefetch_size": 2, "resident_layers": 0}},
    {"layerwise_offload": {"prefetch_size": 1, "resident_layers": 1}},
    {"layerwise_offload": {"prefetch_size": 1, "resident_layers": 0, "extra": 1}},
])
def test_profile_rejects_unmodeled_or_coerced_settings(changes):
    with pytest.raises(ValueError):
        workflow_module.resolve_config(profile_config(**changes))


def test_no_prefetch_variant_and_legacy_precision_are_explicit():
    _, _, resolved = workflow_module.resolve_config(profile_config(
        layerwise_offload={"prefetch_size": 0, "resident_layers": 0}))
    assert resolved["layerwise_offload"]["prefetch_size"] == 0
    with pytest.raises(ValueError, match="compute_dtype=fp16"):
        workflow_module.resolve_config({"compute_dtype": "bf16"})
    with pytest.raises(ValueError, match="requires execution_profile"):
        workflow_module.resolve_config({"layerwise_offload": {
            "prefetch_size": 1, "resident_layers": 0}})


def test_official_defaults_count_updates_and_do_not_claim_stage_unload():
    config = profile_config()
    del config["num_inference_steps"]
    cfg, _, resolved = workflow_module.resolve_config(config)
    assert cfg.num_inference_steps == 49
    assert resolved["assume_stage_unload"] is False
    workflow = workflow_module.create_workflow(profile_hardware(), config)
    assert workflow.metadata["stage_unload_applied"] is False
