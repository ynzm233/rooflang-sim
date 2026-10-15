# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Shared CSV schema and model-specific export projections.

Workflows provide simulation facts. This module decides which of those facts
belong in the public CSV interface and keeps their column order stable.
"""

from collections.abc import Mapping, Sequence


EXPORT_SCHEMA_VERSION = 2

OPERATOR_FIELDS = (
    "model_name node_name operator op_type shape hardware_name "
    "execution_time_ms device_ms compute_ms memory_ms network_ms "
    "memory_traffic_bytes compute_dtype weight_storage_dtype communicate_dtype "
    "activation weight_read weight_saved flops op_backend "
    "activation_peak_bytes_proxy graph_label graph_multiplier row_kind device "
    "timing_status timing_included device_names stream start_ms end_ms bound "
    "flops_by_dtype"
).split()

COMMON_OPERATOR_EXTENSIONS = (
    "tag_mask kv_storage_dtype kv_cache_read kv_cache_saved kv_cache_load "
    "overlapped_compute_ms overlapped_memory_ms resident_weight_bytes "
    "cpu_flops cpu_flops_breakdown"
).split()

STAGE_FIELDS = (
    "name execution_time_ms compute_time_ms memory_time_ms comm_time_ms "
    "total_flops total_mem_bytes total_weight_bytes peak_activation_bytes "
    "invocations device timing_status cpu_total_flops cpu_weight_bytes "
    "cpu_peak_memory_bytes_proxy cpu_execution_time_ms "
    "measurement_start_ms measured_execution_time_ms peak_memory_bytes "
    "operator_count"
).split()

PLAN_FIELDS = (
    "model_name export_schema_version chip_name stages batch_size prompt_len total_cards "
    "main_cp main_dp main_ep main_pp compute_dtype weight_storage_dtype "
    "peak_stage_memory_bytes fits_peak_stage_memory execution_time_ms "
    "measured_execution_time_ms throughput request_per_second "
    "is_valid fail_reason"
).split()

COMMON_PLAN_EXTENSIONS = (
    "pp_partition measurement_start output_tokens request_count "
    "workload operator_stage_csvs"
).split()

LLM_MODELS = frozenset(("dsv4_pro", "dsv4_flash", "glm_5_2_fp8", "kimi_k3"))

MAGI_OPERATOR_EXTENSIONS = (
    "module layer_idx block_id semantic assumption source_compute_dtype "
    "component rank_scope memory_model parent_node_name subcall subcall_index "
    "traffic_scope source_weight_saved recipe_node_index graph_node_index"
).split()

MAGI_PLAN_EXTENSIONS = (
    "workflow resolution negative_text_tokens num_diffusion_steps refiner_steps "
    "resolved_shape_summary dit_tokens condition_type cfg thw_list "
    "vae_downsampling_factor patch_downsampling_factor kv_storage_dtype "
    "deploy_strategy main_chip main_cards main_concurrency "
    "main_tp main_sp main_moe_dp main_moe_tp main_mlp_tp main_oproj_tp "
    "compute_scope timing_status end_to_end_timing_complete timing_backend "
    "memory_status capacity_verdict_is_provisional preview_forward_count "
    "refiner_forward_count all_resident_memory_bytes fits_all_resident_memory "
    "memory_accounting operator_graph_stages operator_graph_multipliers"
).split()

MODEL_EXTENSIONS = {
    **{name: {"operator": (), "stage": (), "plan": ()} for name in LLM_MODELS},
    "magi2_preview": {
        "operator": MAGI_OPERATOR_EXTENSIONS,
        "stage": (),
        "plan": MAGI_PLAN_EXTENSIONS,
    },
}


def _meaningful(value: object) -> bool:
    """Zero and False are real measurements; only missing text is absent."""
    return value is not None and (not isinstance(value, str) or bool(value))


def csv_fields(kind: str, model_name: str, rows: Sequence[Mapping]) -> list[str]:
    """Return ordered, applicable columns for one CSV file.

    Base operator/stage fields remain available even for an empty trace. Their
    extensions and optional plan fields require at least one actual value.
    Unknown model names retain the generic API's open metadata extension point.
    """
    base = {"operator": OPERATOR_FIELDS, "stage": STAGE_FIELDS,
            "plan": PLAN_FIELDS}[kind]
    declared = MODEL_EXTENSIONS.get(model_name)
    common = {"operator": COMMON_OPERATOR_EXTENSIONS,
              "stage": (), "plan": COMMON_PLAN_EXTENSIONS}[kind]
    extension = list(common) + (
        list(declared[kind]) if declared is not None else [])
    present = {key for row in rows for key, value in row.items()
               if _meaningful(value)}
    result = []
    for field in base:
        if kind == "plan" and field in ("throughput", "request_per_second"):
            # A count is required for a rate. A known count of zero is valid.
            count = "output_tokens" if field == "throughput" else "request_count"
            if count not in present:
                continue
        result.append(field)
    result.extend(field for field in extension
                  if field not in result and field in present)
    if declared is None:
        result.extend(key for row in rows for key in row
                      if key not in result and key in present and not key.startswith("_"))
    return result


def _csv_tuple(values: tuple[int, ...]) -> str:
    return "(" + ",".join(map(str, values)) + ")"


def project_plan_metadata(model_name: str, hardware_name: str,
                          metadata: Mapping | None) -> dict:
    """Project private model facts into the public plan schema."""
    source = dict(metadata or {})
    config = source.pop("_model_config", None)
    plan = {key: value for key, value in source.items() if not key.startswith("_")}
    if model_name == "magi2_preview" and config is not None:
        shape = plan["resolved_shape_summary"]
        plan.update({
            "dit_tokens": shape["preview_padded_tokens"],
            "condition_type": config.condition_type,
            "cfg": config.cfg,
            "thw_list": ";".join(_csv_tuple(dims) for dims in config.thw_list),
            "vae_downsampling_factor": _csv_tuple(config.vae_downsampling_factor),
            "patch_downsampling_factor": _csv_tuple(config.patch_downsampling_factor),
            "kv_storage_dtype": "bf16",
            "deploy_strategy": "magi2_preview",
            "main_chip": hardware_name,
            "main_cards": config.cp_size,
            "main_concurrency": 1,
            "main_tp": 1,
            "main_sp": 1,
            "main_moe_dp": 1,
            "main_moe_tp": 1,
            "main_mlp_tp": 1,
            "main_oproj_tp": 1,
        })
    elif "main_chip" in plan and not plan["main_chip"]:
        plan["main_chip"] = hardware_name
    if model_name in MODEL_EXTENSIONS:
        allowed = (set(PLAN_FIELDS) | set(COMMON_PLAN_EXTENSIONS)
                   | set(MODEL_EXTENSIONS[model_name]["plan"]))
        plan = {key: value for key, value in plan.items() if key in allowed}
    return plan
