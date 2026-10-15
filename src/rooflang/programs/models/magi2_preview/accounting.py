"""Source-derived MAGI-2 accounting, independent of timing/operator graphs.

These are analytical payload/working-set proxies, not allocator measurements.
The checkpoint indices expose only total bytes and names, not per-tensor dtype
or storage metadata.  Runtime dtype conversion and compiler buffers therefore
remain explicit uncertainties in the indexed Preview/Refiner proxies.
"""

from __future__ import annotations

from math import ceil, prod
from typing import Any, Mapping

from rooflang.programs.models.magi2_preview.config import load_model_spec, model_architecture

# Ported from Markov_Infer_Sim/src/application/magi2_accounting.py
# Source SHA256: 7e0e8bb4df5a1985da5b7bc0e6e1d7d54cb4d6a85517100cc2fca996a5284e67


SOURCE_COMMIT = "f68a0f9"
SOURCE_BASE = "https://github.com/SandAI-org/MAGI-2-preview/blob/" + SOURCE_COMMIT


def _spec(model_config: Any = None) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    raw = load_model_spec(model_config)
    return model_architecture(raw), raw["indexed_weight_bytes"]


def _bytes(dtype: str) -> int:
    return {"bf16": 2, "fp16": 2, "fp32": 4, "int32": 4, "int64": 8}[dtype]


def _qwen_dimensions(config: Mapping[str, Any]) -> tuple[int, ...]:
    return (config["hidden_size"], config["intermediate_size"],
            config["num_attention_heads"] * config["head_dim"],
            config["num_kv_heads"] * config["head_dim"],
            config["linear_num_key_heads"] * config["linear_key_head_dim"],
            config["linear_num_value_heads"] * config["linear_value_head_dim"],
            config["linear_num_value_heads"])


def _get(shape: Any, key: str, default: Any = None) -> Any:
    return shape.get(key, default) if isinstance(shape, Mapping) else getattr(shape, key, default)


def _tensor_bytes(shape: Any, key: str, element_bytes: int = 4) -> int:
    dims = _get(shape, key, ())
    return int(prod(dims) * element_bytes) if dims else 0


def qwen_text_parameter_count(config: Mapping[str, Any] | None = None) -> int:
    """Actual Qwen3_5TextModel: no vision, LM head or MTP predictor."""
    config = config or _spec()[0]["qwen_text"]
    hidden, ffn, q, kv, key, value, heads = _qwen_dimensions(config)
    full = hidden * (2 * q + 2 * kv) + q * hidden + 2 * config["head_dim"]
    linear = hidden * (2 * key + 3 * value + 2 * heads)
    linear += (2 * key + value) * config["linear_conv_kernel_size"] + 2 * heads + config["linear_value_head_dim"]
    return (config["vocabulary_size"] * hidden + config["num_layers"] * (3 * hidden * ffn + 2 * hidden)
            + config["num_full_attention_layers"] * full + config["num_linear_attention_layers"] * linear + hidden)


def _qwen_forward(length: int, config: Mapping[str, Any]) -> dict[str, int]:
    """Major mathematical FLOPs for the CPU PyTorch GDN fallback."""
    if not 0 < length <= config["max_tokens"]:
        raise ValueError(f"Qwen encoded sequence length must be in [1, {config['max_tokens']}]")
    hidden, ffn, q, kv, key, value, heads = _qwen_dimensions(config)
    chunk = config["linear_chunk_size"]
    dk, dv = config["linear_key_head_dim"], config["linear_value_head_dim"]
    full_layers, linear_layers = config["num_full_attention_layers"], config["num_linear_attention_layers"]
    conv_kernel = config["linear_conv_kernel_size"]
    weight_size, recurrence_size = _bytes(config["weight_dtype"]), _bytes(config["recurrence_dtype"])
    chunks = ceil(length / chunk)
    flops = {
        "ffn_matmul_flops": config["num_layers"] * 6 * length * hidden * ffn,
        "linear_attention_projection_flops": linear_layers * 2 * length * hidden * (2 * key + 3 * value + 2 * heads),
        "full_attention_projection_flops": full_layers * 2 * length * hidden * (3 * q + 2 * kv),
        "causal_attention_matmul_flops": full_layers * 4 * config["num_attention_heads"] * config["head_dim"] * (length * (length + 1) // 2),
        # Conv1d padding=3 first materializes L+3 positions, then crops.
        "depthwise_conv_flops": linear_layers * 2 * (length + conv_kernel - 1) * (2 * key + value) * conv_kernel,
        "gdn_chunk_matmul_flops": linear_layers * heads * chunks * (
            2 * chunk * chunk * dk * 3
            + 2 * chunk * chunk * dv * 2
            + 2 * chunk * dk * dv * 3
        ),
        "gdn_triangular_recurrence_scalar_flops": linear_layers * heads * chunks * sum(2 * i * i for i in range(1, chunk)),
    }
    flops["major_flops"] = sum(flops.values())
    # Cache lasts for one encode call. Positive/negative calls do not share it.
    kv_bytes = full_layers * 2 * kv * length * weight_size
    recurrent_bytes = linear_layers * heads * dk * dv * recurrence_size
    conv_cache_bytes = linear_layers * (2 * key + value) * conv_kernel * weight_size
    # output_hidden_states=True: 65 logical states. This is a storage upper
    # proxy for that history only; final hidden output may alias a history entry.
    hidden_history_bytes = (config["num_layers"] + 1) * length * hidden * weight_size
    return {
        "tokens": length,
        "tokenizer_padded_tokens": length,
        "gdn_padded_tokens": chunks * chunk,
        **flops,
        "full_attention_dense_matmul_flops_upper": full_layers * 4 * config["num_attention_heads"] * config["head_dim"] * length * length,
        "kv_cache_bytes": kv_bytes,
        "gdn_recurrent_cache_bytes": recurrent_bytes,
        "gdn_conv_cache_bytes": conv_cache_bytes,
        "hidden_history_bytes_proxy": hidden_history_bytes,
        "cache_and_hidden_history_bytes_proxy": kv_bytes + recurrent_bytes + conv_cache_bytes + hidden_history_bytes,
        "output_embedding_bytes": length * hidden * weight_size,
    }


def cpu_text_accounting(shape: Any, model_config: Any = None) -> dict[str, Any]:
    """Return CPU work and memory without inventing a CPU execution time."""
    architecture, indexed = _spec(model_config)
    config = architecture["qwen_text"]
    lengths = [int(_get(shape, "positive_text_tokens")), int(_get(shape, "negative_text_tokens", 0))]
    calls = [_qwen_forward(length, config) for length in lengths if length > 0]
    parameters = qwen_text_parameter_count(config)
    weights = parameters * _bytes(config["weight_dtype"])
    cache_peak = max((call["cache_and_hidden_history_bytes_proxy"] for call in calls), default=0)
    return {
        "device": "cpu",
        "rank_scope": "CP leader only",
        "forward_calls": len(calls),
        "invocations": len(calls),
        "execution_time_ms": None,
        "included_in_gpu_time": False,
        "total_flops": sum(call["major_flops"] for call in calls),
        "major_flops": sum(call["major_flops"] for call in calls),
        "parameter_count": parameters,
        "weight_bytes": weights,
        "resident_weight_bytes": weights,
        "total_weight_bytes": weights,
        "checkpoint_index_total_bytes": indexed["qwen_text"],
        "cache_and_hidden_history_peak_bytes_proxy": cache_peak,
        "peak_memory_bytes_proxy": weights + cache_peak,
        "embedding_payload_bytes": sum(call["output_embedding_bytes"] for call in calls),
        "calls": calls,
        "sources": [SOURCE_BASE + "/inference/model/qwen35.py", "offline_pack/sources/model/text_encoder/config.json", "offline_pack/sources/runtime/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py"],
        "assumptions": [
            f"Single prompt per call, right padding='longest', truncation at {config['max_tokens']}; no fixed maximum-length compute padding.",
            "Positive and negative prompts have separate full forwards. I2V Figure-token pooling reuses the positive embedding and adds no model forward.",
            f"skip_layer={config['skip_layer']} selects a captured state after the full {config['num_layers']}-layer forward; it does not skip the last layers.",
            "Major FLOPs include linear/MLP/full causal attention and the unfused CPU GDN fallback; nonlinear/norm/tokenizer operations are not folded into invented FLOPs.",
            f"CPU runtime/kernel availability is not validated. GDN assumes PyTorch fallback, {config['linear_chunk_size']}-token chunks and {config['recurrence_dtype']} recurrence.",
            "BF16 parameter bytes derive from TextModel constructors; full checkpoint also contains vision and output-head tensors that TextModel does not load.",
            f"Cache/history proxy excludes fallback temporaries, allocator overhead and libraries; {config['num_layers']+1} logical hidden states can share storage. CPU timing remains unknown.",
        ],
    }


def preview_weight_accounting(ep_size: int = 8, model_config: Any = None) -> dict[str, Any]:
    """Shard only expert/router tensors; retain replicated non-expert weights."""
    if ep_size < 1:
        raise ValueError("ep_size must be positive")
    architecture, indexed = _spec(model_config)
    config = architecture["preview_transformer"]
    layers = config["num_layers"] - len(config["dense_ffn_layers"])
    heads, experts = config["num_moe_heads"], config["experts_per_head"]
    dim, expert_ffn = config["expert_head_dim"], config["expert_hidden_size"]
    weight_size, router_size = _bytes(config["weight_dtype"]), _bytes(config["router_dtype"])
    local_heads = ceil(heads / ep_size)
    routed_global = layers * heads * experts * 3 * dim * expert_ffn * weight_size
    router_global = layers * (heads * experts * dim + 2 * heads * experts) * router_size
    replicated_proxy = indexed["preview_transformer"] - routed_global - router_global
    routed_local = layers * local_heads * experts * 3 * dim * expert_ffn * weight_size
    router_local = layers * (local_heads * experts * dim + 2 * local_heads * experts) * router_size
    pad_buffers = layers * (local_heads * ep_size - heads) * dim * weight_size
    if replicated_proxy < 0:
        raise ValueError("preview index payload is smaller than configured sharded expert weights")
    return {
        "checkpoint_index_total_bytes": indexed["preview_transformer"],
        "routed_experts_global_bytes": routed_global,
        "router_and_bias_global_bytes_assuming_fp32": router_global,
        "replicated_nonexpert_bytes_proxy": replicated_proxy,
        "local_heads_allocated": local_heads,
        "padded_heads": local_heads * ep_size,
        "active_expert_ranks": ceil(heads / local_heads),
        "routed_experts_per_rank_bytes": routed_local,
        "router_and_bias_per_rank_bytes": router_local,
        "ep_pad_buffer_per_rank_bytes": pad_buffers,
        "resident_per_rank_bytes_proxy": replicated_proxy + routed_local + router_local + pad_buffers,
        "formula": "index_total - global_experts_BF16 - global_router_FP32 + local_experts_BF16 + local_router_FP32 + nonpersistent_EP_pad_zeros",
        "status": "indexed-payload-derived proxy; runtime casts/other nonpersistent buffers excluded",
        "notes": [
            f"EP{ep_size} pads {heads} to {local_heads * ep_size} heads; every rank allocates {local_heads} heads, including dummy-head ranks.",
            "Dense/shared/adapters are replicated; CP does not shard their weights.",
            "Router gate and two bias buffers are FP32 in the constructor. Applying that dtype to index subtraction is an explicit assumption without safetensors headers.",
        ],
    }


def vae_weight_accounting(model_config: Any = None) -> dict[str, Any]:
    """Whole loaded modules, counted once regardless of executed decoder windows."""
    architecture, _ = _spec(model_config)
    wan, turbo, audio = (architecture[name] for name in
                         ("image_vae_encoder", "video_vae_decoder", "audio_vae_decoder"))
    wan_params, audio_params = wan["resident_parameters"], audio["resident_parameters"]
    wan_encoder, wan_decoder, wan_bottleneck = (wan_params[name] for name in ("encoder", "decoder", "bottleneck"))
    audio_encoder, audio_decoder = (audio_params[name] for name in ("encoder", "decoder"))
    turbo_parameters = turbo["resident_parameters"]["decoder"]
    return {
        "wan_encoder_parameters": wan_encoder,
        "wan_decoder_parameters": wan_decoder,
        "wan_bottleneck_parameters": wan_bottleneck,
        "wan_full_vae_fp32_bytes": _bytes(wan["weight_dtype"]) * (wan_encoder + wan_decoder + wan_bottleneck),
        "wan_scale_tensor_bytes": 3 * wan["latent_channels"] * 4,
        "turbo_decoder_parameters": turbo_parameters,
        "turbo_decoder_bf16_bytes": turbo_parameters * _bytes(turbo["weight_dtype"]),
        "turbo_scale_tensor_bytes": 3 * turbo["latent_channels"] * 4,
        "audio_encoder_parameters": audio_encoder,
        "audio_decoder_parameters": audio_decoder,
        "audio_full_autoencoder_fp32_bytes": (audio_encoder + audio_decoder) * _bytes(audio["weight_dtype"]),
        "audio_scale_tensor_bytes": 8,
        "audio_decoder_weight_norm_materialized_bytes_proxy": audio_params["materialized_decoder_weight"] * _bytes(audio["weight_dtype"]),
        "notes": [
            "Wan2.2 full VAE is FP32 and resident on all ranks, including T2V. Encoder and decoder graph weights must not be added again.",
            "Turbo decoder and full audio autoencoder are resident only on the CP leader. Audio loads idle encoder parameters as well as decoder parameters.",
            "Audio counts include weight_v/weight_g and Snake alpha/beta. Legacy weight_norm also keeps a materialized decoder weight after execution; a warmed-request proxy includes that duplicate.",
            "Wan/Turbo scale lists retain mean/std/inverse-std tensors. Constructor counts exclude compiler and library workspaces.",
            "Resident parameter facts are local architecture checkpoint-specific values; changing codec topology requires updating those full-module facts, including idle submodules.",
        ],
    }


def memory_accounting(
    shape: Any,
    cp_size: int = 8,
    ep_size: int = 8,
    stage_activation_bytes: Mapping[str, int] | None = None,
    assume_stage_unload: bool = True,
    model_config: Any = None,
) -> dict[str, Any]:
    """Conservative named working-set scenario, never an allocator-fit proof.

    Caller activations are graph single-op maxima. Known full-pack/latent/output
    lifetimes are accounted separately; max() avoids blindly adding the same
    graph allocation to a larger explicitly modeled temporary. Scheduler/CFG
    multiplicities are scenarios and are deliberately exposed in the result.
    """
    if cp_size < 1 or ep_size < 1:
        raise ValueError("cp_size and ep_size must be positive")
    if cp_size != ep_size:
        raise ValueError("The released overlapping CP/EP mapping requires cp_size == ep_size")
    activations = {key: max(0, int(value)) for key, value in (stage_activation_bytes or {}).items()}
    architecture, indexed = _spec(model_config)
    preview = preview_weight_accounting(ep_size, model_config)
    vaes = vae_weight_accounting(model_config)
    preview_weight = preview["resident_per_rank_bytes_proxy"]
    refiner_active = bool(_get(shape, "refiner_forward_calls", _get(shape, "refiner_video_tokens", 1)))
    refiner_weight = indexed["refiner_transformer"] if refiner_active else 0
    common = vaes["wan_full_vae_fp32_bytes"] + vaes["wan_scale_tensor_bytes"]
    leader_extra = (vaes["turbo_decoder_bf16_bytes"] + vaes["turbo_scale_tensor_bytes"]
                    + vaes["audio_full_autoencoder_fp32_bytes"] + vaes["audio_scale_tensor_bytes"]
                    + vaes["audio_decoder_weight_norm_materialized_bytes_proxy"])
    leader_idle = common + leader_extra
    p_latent = _tensor_bytes(shape, "preview_latent_shape")
    r_latent = _tensor_bytes(shape, "refiner_latent_shape")
    audio = _tensor_bytes(shape, "audio_latent_shape")
    image = _tensor_bytes(shape, "image_latent_shape")
    text_dim = architecture["qwen_text"]["hidden_size"]
    text_size = _bytes(architecture["qwen_text"]["weight_dtype"])
    text = (int(_get(shape, "positive_text_tokens", 0)) + int(_get(shape, "negative_text_tokens", 0))) * text_dim * text_size
    text += text_dim * text_size if int(_get(shape, "image_tokens", 0)) else 0
    p_pack = int(_get(shape, "preview_padded_tokens", 0)) * text_dim * 4
    r_rows = max(int(_get(shape, "refiner_cond_tokens", 0)), int(_get(shape, "refiner_uncond_tokens", 0)))
    r_pack = r_rows * text_dim * 4
    output_f32 = _tensor_bytes(shape, "decoded_video_shape")
    output_bf16 = output_f32 // 2
    # Explicit scenarios: original+CFG/predictions+scheduler histories, plus
    # packed modality temporaries overlapping cat. These are not hard bounds.
    persistent = {
        "text_embeddings_bf16_bytes": text,
        "preview_latent_fp32_bytes": p_latent,
        "refiner_latent_fp32_bytes": r_latent,
        "audio_latent_fp32_bytes": audio,
        "reference_latent_fp32_bytes": image,
        "preview_full_packed_fp32_bytes": p_pack,
        "refiner_full_packed_fp32_bytes": r_pack,
        "decoded_video_bf16_bytes": output_bf16,
        "decoded_video_fp32_bytes": output_f32,
    }
    stage_extras = {
        "condition_broadcast": text,
        "text_broadcast": text,
        "image_broadcast": text + image,
        "image_vae_encode": image,
        "preview_denoise": text + image + 8 * (p_latent + audio),
        "refiner_denoise": text + image + 2 * p_latent + audio + 8 * r_latent,
        "latent_upsample": text + image + p_latent + audio,
        "video_vae_decode": text + image + p_latent + r_latent + audio,
        "audio_vae_decode": text + image + p_latent + r_latent + audio,
    }
    temporaries = {
        "condition_broadcast": text,
        "text_broadcast": text,
        "image_broadcast": image,
        "image_vae_encode": 0,
        "preview_denoise": 2 * p_pack,
        "refiner_denoise": 2 * r_pack,
        "latent_upsample": r_latent,
        # cat has chunk outputs and a fresh full output; postprocess float
        # conversions/multiply/add/clamp can overlap multiple full outputs.
        "video_vae_decode": max(2 * output_bf16, 3 * output_f32),
        "audio_vae_decode": output_f32,
    }
    if not refiner_active:
        stage_extras.pop("refiner_denoise")
        temporaries.pop("refiner_denoise")
        stage_extras.pop("latent_upsample")
        temporaries.pop("latent_upsample")
    stages = {}
    stage_names = list(activations) if activations else list(stage_extras)
    if not activations:
        stage_names.remove("condition_broadcast")  # legacy alias, not an additional stage
        if not int(_get(shape, "image_tokens", 0)):
            stage_names = [name for name in stage_names if name not in {"image_broadcast", "image_vae_encode"}]
    for name in stage_names:
        active_weight = preview_weight if name in {"condition_broadcast", "text_broadcast", "image_broadcast", "image_vae_encode", "preview_denoise"} else (refiner_weight if name == "refiner_denoise" else 0)
        if not assume_stage_unload:
            active_weight = preview_weight + refiner_weight
        working = max(activations.get(name, 0), temporaries.get(name, 0))
        stages[name] = {
            "gpu_resident_weight_bytes_proxy": leader_idle + active_weight,
            "persistent_and_scheduler_bytes_proxy": stage_extras.get(name, 0),
            "graph_peak_activation_bytes": activations.get(name, 0),
            "explicit_temporary_bytes_proxy": temporaries.get(name, 0),
            "memory_bytes_proxy": leader_idle + active_weight + stage_extras.get(name, 0) + working,
        }
    all_weights = leader_idle + preview_weight + refiner_weight
    peak = max((v["memory_bytes_proxy"] for v in stages.values()), default=all_weights)
    all_resident = all_weights + max((v["persistent_and_scheduler_bytes_proxy"] + max(v["graph_peak_activation_bytes"], v["explicit_temporary_bytes_proxy"]) for v in stages.values()), default=0)
    cpu = cpu_text_accounting(shape, model_config)
    return {
        "rank_scope": "CP leader / worst loaded weight rank; all values are per rank unless named total",
        "status": "analytical working-set proxy, not allocator measurement or capacity guarantee",
        "capacity_verdict_is_provisional": True,
        "architecture_source": "model_config.components",
        "indexed_weight_source": "model_config.indexed_weight_bytes",
        "assume_stage_unload": assume_stage_unload,
        "peak_stage_memory_bytes": peak,
        "all_resident_memory_bytes": all_resident,
        "all_resident_weight_bytes": all_weights,
        "leader_idle_gpu_resident_bytes_proxy": leader_idle,
        "nonleader_idle_gpu_resident_bytes_proxy": common,
        "preview": preview,
        "refiner_resident_per_rank_bytes_proxy": refiner_weight,
        "vae": vaes,
        "persistent_tensors": persistent,
        "scenario_factors": {"scheduler_and_cfg_latent_copies": 8, "pack_temporary_copies": 2, "postprocess_float_output_copies": 3},
        "stages": stages,
        "latent_upsample_weight_state": "Preview offloaded; Refiner loaded after interpolation under stage-unload scenario",
        "host_leader_weight_bytes_proxy": preview_weight + refiner_weight + cpu["resident_weight_bytes"],
        "host_all_ranks_weight_bytes_proxy": cp_size * (preview_weight + refiner_weight) + cpu["resident_weight_bytes"],
        "request_weight_h2d_per_rank_bytes_proxy": preview_weight + refiner_weight if assume_stage_unload else 0,
        "request_weight_d2h_bytes": 0,
        "sources": [SOURCE_BASE + "/inference/pipeline/inference_engine.py", SOURCE_BASE + "/inference/model/magi2_preview.py", SOURCE_BASE + "/inference/common/module_state_backup.py", SOURCE_BASE + "/inference/model/vae2_2.py", SOURCE_BASE + "/inference/model/turbo_vaed.py", SOURCE_BASE + "/inference/model/sa_audio_vae.py"],
        "assumptions": [
            "Default CPU text encoder, roundtrip Preview/Refiner, GPU resident VAEs; Preview is unloaded before Refiner loads. CPU backups persist and restoring their storage does not copy GPU weights back to CPU.",
            "Preview weight estimate uses checkpoint totals plus expert-padding/sharding arithmetic; Refiner dense checkpoint bytes are replicated, never divided by CP.",
            "Eight latent-sized allocations approximate CFG/UniPC/history overlap; two full packed buffers approximate modality padding and concatenation before CP slicing. These configurable-shape scenarios are neither measured peaks nor rigorous upper bounds.",
            "Single-op graph peaks cannot reconstruct allocator lifetimes. This scenario excludes unknown fused-kernel/library/compiler workspaces, fragmentation/reserved memory and cold-start state-dict peaks.",
            "Video output proxy uses actual decoded tensor dimensions, including 1088-row storage at nominal 1080p. CPU postprocessing/mux buffers are separate from GPU capacity.",
            "All-resident is a comparison scenario. CP/EP overlap on the same ranks; do not multiply cp_size by ep_size.",
        ],
    }
