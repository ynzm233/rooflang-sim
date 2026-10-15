"""Model-owned MAGI-2 semantic stages and repeated graph execution.

Schedule adapted from Markov_Infer_Sim/src/application/magi2_app.py.
Geometry, operators and residency are local; timing uses RoofLang's DES.
"""

from functools import partial

from rooflang.language.hardware.component import Compute
from rooflang.programs.experiments.simulation import simulate
from rooflang.programs.experiments.workflow import Workflow
from rooflang.runtime.stages import GraphResult, StageResult

from .accounting import cpu_text_accounting, memory_accounting
from .config import Magi2Config, load_workload


def _resolve_config(overrides):
    overrides = dict(overrides)
    if overrides.pop("batch_size", 1) != 1:
        raise ValueError("MAGI-2 supports one clip per request (batch_size=1)")
    if overrides.pop("dp", 1) != 1:
        raise ValueError("MAGI-2 uses a shared CP/EP mesh; DP must be 1")
    if overrides.pop("visualization", False):
        raise ValueError("MAGI-2 layer visualization is not implemented")
    if overrides.get("precision_proxy", {"bf16_from_fp16": 1.0, "fp32_from_fp16": 0.5}) != {
            "bf16_from_fp16": 1.0, "fp32_from_fp16": 0.5}:
        raise ValueError("RoofLang uses hardware dtype peaks; precision_proxy overrides are unsupported")
    for short in ("cp", "ep"):
        if short in overrides:
            overrides[f"{short}_size"] = overrides.pop(short)
    if "cp_size" in overrides and "ep_size" not in overrides:
        overrides["ep_size"] = overrides["cp_size"]
    defaults = load_workload()
    workflow = overrides.get("workflow", defaults["workflow"])
    profiles = defaults.pop("workflows")
    if workflow not in profiles:
        raise ValueError("MAGI-2 workflow must be i2v or t2v")
    return Magi2Config.from_dict({**defaults, **profiles[workflow], **overrides})


def create_workflow(hardware, config):
    cfg = _resolve_config(config)
    mm = cfg.runtime_multimodal_config()
    shape = mm["shape_summary"]
    gpu_nodes = [node for node in hardware.nodes
                 if isinstance(node, Compute) and node.kind == "gpu"]
    if len(gpu_nodes) < cfg.cp_size:
        raise ValueError(f"MAGI-2 CP={cfg.cp_size} needs at least {cfg.cp_size} GPU components")
    cpu = cpu_text_accounting(shape)
    # component, graph label, per-graph configuration, calls, distributed
    jobs = {"text_encode": [
        ("condition_broadcast", "text_broadcast", {"broadcast_kind": "text"}, 1, True)]}
    if cfg.workflow == "i2v":
        jobs["image_vae_encode"] = [
            ("image_vae_encoder", "image_vae_encode", {}, 1, False),
            ("condition_broadcast", "image_broadcast", {"broadcast_kind": "image"}, 1, True)]
    jobs["preview_denoise"] = [
        ("preview_transformer", "preview_denoise", {}, cfg.preview_steps, True)]
    if cfg.resolution == "1080p":
        jobs["refiner_denoise"] = [
            ("latent_upsample", "latent_upsample", {}, 1, False),
            *[("refiner_transformer", f"refiner_{branch}", {"branch": branch},
               cfg.refiner_steps, True) for branch in ("cond", "uncond")]]
    jobs["video_vae_decode"] = [
        ("video_vae_decoder", f"video_decode_window_{index:02d}",
         {"turbo_window_index": index}, 1, False)
        for index in range(shape["turbo_windows"]["calls"])]
    jobs["audio_vae_decode"] = [("audio_vae_decoder", "audio_vae_decode", {}, 1, False)]

    def run_stage(name):
        from .bridge import build_component
        stage = StageResult()
        for component, label, updates, multiplier, distributed in jobs[name]:
            graph, placement = build_component(
                component, {**mm, **updates}, cfg.architecture, hardware,
                cp=cfg.cp_size if distributed else 1,
                ep=cfg.ep_size if distributed else 1)
            result = simulate(graph, placement, hardware)
            stage.graphs.append(GraphResult(label, result, multiplier))
        stage.metrics["timing_status"] = "modeled_accelerator_operators"
        if name == "text_encode":
            for branch, call in zip(("positive", "negative"), cpu["calls"]):
                stage.untimed_rows.append({
                    "node_name": f"Qwen_text_{branch}", "operator": "QwenTextModel",
                    "shape": f"tokens={call['tokens']}", "device": "cpu",
                    "row_kind": "cpu_forward_accounting",
                    "graph_label": f"qwen_{branch}", "cpu_flops": call["major_flops"],
                    "cpu_flops_breakdown": {
                        key: value for key, value in call.items()
                        if key.endswith("_flops") and key != "major_flops"
                    },
                    "assumption": "Full CPU text forward; latency and temporary traffic uncalibrated",
                })
            stage.metrics.update({
                "timing_status": "partial_cpu_untimed",
                "cpu_weight_bytes": cpu["resident_weight_bytes"],
                "cpu_peak_memory_bytes_proxy": cpu["peak_memory_bytes_proxy"],
            })
        return stage

    def summarize(results):
        activations, phase_stages = {}, {}
        for name, stage in results.items():
            for graph in stage.graphs:
                phase = graph.label if graph.label in (
                    "text_broadcast", "image_broadcast", "latent_upsample") else name
                phase_stages[phase] = name
                peak = max((entry.kernel.csv_metadata.get("activation_peak_bytes_proxy", 0)
                            for entry in graph.result.trace
                            if hasattr(entry.kernel, "csv_metadata")), default=0)
                activations[phase] = max(activations.get(phase, 0), peak)
        memory = memory_accounting(shape, cfg.cp_size, cfg.ep_size, activations,
                                   cfg.assume_stage_unload)
        # Group sequential memory phases only after applying loading/unloading.
        for phase, values in memory["stages"].items():
            name = phase_stages.get(phase)
            if name is not None:
                metrics = results[name].metrics
                metrics["peak_memory_bytes"] = max(
                    metrics.get("peak_memory_bytes", 0), values["memory_bytes_proxy"])
        capacity = min(hardware.find_local_memory(gpu).capacity_gb * 1e9
                       for gpu in gpu_nodes[:cfg.cp_size])
        peak = max((stage.metrics.get("peak_memory_bytes", 0)
                    for stage in results.values()), default=0)
        fits_peak = peak <= capacity
        fits_all = memory["all_resident_memory_bytes"] <= capacity
        valid = fits_peak if cfg.assume_stage_unload else fits_all
        return {
            "preview_forward_count": cfg.preview_steps if "preview_denoise" in results else 0,
            "refiner_forward_count": 2 * cfg.refiner_steps if "refiner_denoise" in results else 0,
            "peak_stage_memory_bytes": peak,
            "all_resident_memory_bytes": memory["all_resident_memory_bytes"],
            "fits_peak_stage_memory": fits_peak, "fits_all_resident_memory": fits_all,
            "is_valid": valid, "fail_reason": "" if valid else "Analytical resident memory exceeds HBM capacity",
            "memory_accounting": memory,
            "operator_graph_stages": {g.label: name for name, stage in results.items() for g in stage.graphs},
            "operator_graph_multipliers": {g.label: g.multiplier for stage in results.values() for g in stage.graphs},
            "request_count": 1 if set(results) == set(jobs) else None,
        }

    return Workflow({name: partial(run_stage, name) for name in jobs}, {
        "workflow": cfg.workflow, "resolution": cfg.resolution,
        "batch_size": 1, "prompt_len": shape["positive_text_tokens"],
        "negative_text_tokens": shape["negative_text_tokens"],
        "total_cards": cfg.cp_size, "main_cp": cfg.cp_size, "main_ep": cfg.ep_size,
        "main_dp": 1, "main_pp": 1, "compute_dtype": "bf16",
        "weight_storage_dtype": "bf16", "num_diffusion_steps": cfg.preview_steps,
        "refiner_steps": cfg.refiner_steps, "resolved_shape_summary": shape,
        "_model_config": cfg,
        "compute_scope": "critical_rank_per_stage_not_global_device_sum",
        "timing_status": "partial_cpu_and_postprocess_untimed",
        "end_to_end_timing_complete": False,
        "timing_backend": "rooflang_discrete_event_simulator",
        "memory_status": "analytical_working_set_proxy_not_measured_allocator_peak",
        "capacity_verdict_is_provisional": True,
    }, summarize)
