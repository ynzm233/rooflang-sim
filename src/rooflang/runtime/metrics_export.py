# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Export operator, stage and plan CSVs from existing simulation results.

Field names follow Markov_Infer_Sim where the metrics have equivalents.
Times are milliseconds, traffic and storage are bytes, throughput is tokens/s.
No simulation or scheduling behavior is changed by this exporter.
"""

from __future__ import annotations

import csv
import json
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from rooflang.language.kernels.comm import CommKernel
from rooflang.runtime.export_config import (
    EXPORT_SCHEMA_VERSION, OPERATOR_FIELDS, PLAN_FIELDS, STAGE_FIELDS,
    csv_fields, project_plan_metadata,
)
from rooflang.runtime.simulator import SimulationResult
from rooflang.runtime.stages import GraphResult, StageResult
from rooflang.runtime.trace_export import _compute_profile, _trace_shape


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {field: row.get(field) for field in fields}
            for field, value in values.items():
                if isinstance(value, (dict, list, tuple)):
                    values[field] = json.dumps(value, ensure_ascii=False)
            writer.writerow(values)


def _device_kind(device):
    return "cpu" if device.kind == "cpu" else "accelerator"


def _dtypes(tensors):
    return "+".join(sorted({tensor.dtype for tensor in tensors}))


def _operator_rows(result, stage, model_name, hardware_name):
    # Collectives have a trace entry on every participant, but represent one
    # kernel whose buffer properties already describe the whole collective.
    entries = defaultdict(list)
    for entry in result.trace:
        entries[entry.kernel].append(entry)
    for index, (kernel, participants) in enumerate(entries.items()):
        entry = participants[0]
        dtype, _, flops_by_dtype = _compute_profile(kernel, entry.device)
        is_comm = isinstance(kernel, CommKernel)
        shape = {
            role: {name: {"shape": _trace_shape(tensor), "dtype": tensor.dtype}
                   for name, tensor in getattr(kernel, role).items()}
            for role in ("inputs", "weights", "outputs")
        }
        activation = float(kernel.input_bytes + kernel.output_bytes)
        row = {
            "model_name": model_name,
            "node_name": f"{type(kernel).__name__}_{index}",
            "operator": type(kernel).__name__,
            "op_type": "communication" if is_comm else "compute",
            "shape": shape,
            "hardware_name": hardware_name,
            "execution_time_ms": (entry.end_us - entry.start_us) / 1000,
            "device_ms": entry.local_elapsed_time_us / 1000,
            "compute_ms": entry.compute_time_us / 1000,
            "memory_ms": entry.memory_time_us / 1000,
            "network_ms": entry.network_time_us / 1000,
            "memory_traffic_bytes": activation + kernel.loaded_weight_bytes,
            "compute_dtype": dtype,
            "weight_storage_dtype": _dtypes(kernel.weights.values()),
            "communicate_dtype": _dtypes(
                [*kernel.inputs.values(), *kernel.outputs.values()])
                if is_comm else "",
            "activation": activation,
            "weight_read": float(kernel.loaded_weight_bytes),
            "weight_saved": float(kernel.weight_bytes),
            "flops": float(kernel.flops),
            "op_backend": "roofline",
            "activation_peak_bytes_proxy": float(
                kernel.input_tensor_bytes + kernel.output_bytes)
                if kernel._requires_placement or is_comm else 0.0,
            "graph_label": stage,
            "graph_multiplier": 1,
            "row_kind": "operator",
            "device": "+".join(sorted({_device_kind(e.device)
                                         for e in participants})),
            "timing_status": "modeled",
            "timing_included": True,
            "device_names": "+".join(sorted({e.device.name
                                               for e in participants})),
            "stream": entry.stream,
            "start_ms": entry.start_us / 1000,
            "end_ms": entry.end_us / 1000,
            "bound": entry.bound.value,
            "flops_by_dtype": flops_by_dtype,
        }
        if not is_comm:
            # Ideal roofline bottleneck attribution, before DES contention.
            # Ties belong to both terms; these are not additive wall times.
            row["overlapped_compute_ms"] = (
                row["compute_ms"] if row["compute_ms"] >= row["memory_ms"] else 0.0)
            row["overlapped_memory_ms"] = (
                row["memory_ms"] if row["memory_ms"] >= row["compute_ms"] else 0.0)
        # Model builders supply semantic identity, never replacement timings.
        for key, value in getattr(kernel, "csv_metadata", {}).items():
            if key not in row or key in {
                "node_name", "operator", "op_type", "shape", "compute_dtype",
                "weight_storage_dtype", "communicate_dtype", "activation_peak_bytes_proxy",
            }:
                row[key] = value
        if row.get("source_weight_saved") is not None:
            # Source accounting may charge shared weights only at their owner.
            # Preserve the physical tensor size separately for memory analysis.
            row["resident_weight_bytes"] = float(kernel.weight_bytes)
            row["weight_saved"] = row["source_weight_saved"]
        yield row


def _weight_bytes(result, *, cpu_only=False):
    # Shared weights count once per device. Sparse reads do not shrink the
    # physical storage requirement; weight_read is reported separately.
    weights = {}
    for entry in result.trace:
        if cpu_only and entry.device.kind != "cpu":
            continue
        for tensor in entry.kernel.weights.values():
            key = (entry.device, tensor.weight_id
                   if tensor.weight_id is not None else id(tensor))
            weights[key] = float(tensor.size_bytes)
    return sum(weights.values())


def _stage_row(result, name, rows):
    cpu_rows = [row for row in rows if row["device"] == "cpu"]
    return {
        "name": name,
        # Wall time comes from the DES timeline, not the sum of overlapping
        # operators. Resource-time totals below are additive work estimates.
        "execution_time_ms": result.total_time_us / 1000,
        "compute_time_ms": sum(row["compute_ms"] for row in rows),
        "memory_time_ms": sum(row["memory_ms"] for row in rows),
        "comm_time_ms": sum(row["network_ms"] for row in rows),
        "total_flops": sum(row["flops"] for row in rows),
        "total_mem_bytes": sum(row["memory_traffic_bytes"] for row in rows),
        "total_weight_bytes": _weight_bytes(result),
        "peak_activation_bytes": max(
            (row["activation_peak_bytes_proxy"] for row in rows), default=0),
        "invocations": 1,
        "device": "+".join(sorted({_device_kind(e.device)
                                     for e in result.trace})),
        "timing_status": "modeled",
        "cpu_total_flops": sum(row["flops"] for row in cpu_rows),
        "cpu_weight_bytes": _weight_bytes(result, cpu_only=True),
        "cpu_peak_memory_bytes_proxy": max(
            (row["activation_peak_bytes_proxy"]
             + row.get("resident_weight_bytes", row["weight_saved"])
             for row in cpu_rows), default=0),
        "cpu_execution_time_ms": sum(row["execution_time_ms"]
                                     for row in cpu_rows) if cpu_rows else None,
        "measurement_start_ms": result.measurement_start_us / 1000,
        "measured_execution_time_ms": result.measured_time_us / 1000,
        "peak_memory_bytes": max(result.peak_memory.values(), default=0),
        "operator_count": len(rows),
    }


def _semantic_stage(name, stage, model_name, hardware_name):
    rows, graph_metrics = [], []
    for graph in stage.graphs:
        graph_rows = list(_operator_rows(
            graph.result, graph.label, model_name, hardware_name))
        for row in graph_rows:
            row["graph_multiplier"] = graph.multiplier
        rows.extend(graph_rows)
        graph_metrics.append((_stage_row(graph.result, name, graph_rows), graph.multiplier))
    for extra in stage.untimed_rows:
        rows.append({"model_name": model_name, "hardware_name": hardware_name,
                     "graph_label": name, "graph_multiplier": 1,
                     "row_kind": "unmodeled", "timing_status": "not_estimated",
                     "timing_included": False, "execution_time_ms": None, **extra})
    metrics = _stage_row(SimulationResult([], 0, {}), name, [])
    additive = (
        "execution_time_ms", "compute_time_ms", "memory_time_ms", "comm_time_ms",
        "total_flops", "total_mem_bytes", "measurement_start_ms",
        "measured_execution_time_ms", "cpu_total_flops",
    )
    for key in additive:
        metrics[key] = sum(values[key] * multiplier for values, multiplier in graph_metrics)
    cpu_times = [values["cpu_execution_time_ms"] * multiplier
                 for values, multiplier in graph_metrics
                 if values["cpu_execution_time_ms"] is not None]
    metrics["cpu_execution_time_ms"] = sum(cpu_times) if cpu_times else None
    for key in ("total_weight_bytes", "peak_activation_bytes", "peak_memory_bytes",
                "cpu_weight_bytes", "cpu_peak_memory_bytes_proxy"):
        metrics[key] = max((values[key] for values, _ in graph_metrics), default=0)
    metrics["invocations"] = sum(m for _, m in graph_metrics) + sum(
        row.get("graph_multiplier", 1) for row in stage.untimed_rows)
    metrics["operator_count"] = len(rows)
    metrics["device"] = "+".join(sorted({kind for row in rows
        for kind in row.get("device", "").split("+") if kind}))
    if stage.untimed_rows:
        metrics["timing_status"] = "partial_unmodeled_work"
        if any(row.get("device") == "cpu" for row in stage.untimed_rows):
            metrics["cpu_execution_time_ms"] = None
        metrics["cpu_total_flops"] += sum(
            row.get("cpu_flops", 0) * row.get("graph_multiplier", 1)
            for row in stage.untimed_rows)
    metrics.update(stage.metrics)
    return rows, metrics


def export_metrics(
    stages: Mapping[str, SimulationResult | StageResult], output_dir: str | Path, *,
    model_name: str, hardware_name: str = "", metadata: dict | None = None,
    reproducibility: dict | None = None,
) -> dict:
    """Write one operator CSV per stage, stage_metrics.csv and a plan CSV.

Simple results are one graph invocation; StageResult groups named graphs and
their invocation counts into one semantic stage. Repeated calls reuse resident
weights (the default stage storage proxy is the largest graph's storage).
Models may provide refined stage memory accounting and untimed CPU rows.
Stage/plan execution times include the
full timeline (including preload); measured_execution_time_ms preserves the
simulator's measurement window. Resource totals count unique kernels across
all devices, deduplicating collective participant trace entries. Activation
peaks are per-operator IO proxies; peak_memory_bytes is the largest simulated
per-memory peak. Zero-cost structural aliases do not add activation storage.
total_weight_bytes deduplicates by execution device, not remote/shared memory;
memory feasibility uses the simulator's actual per-memory peaks instead.
Cross-stage resident memory is not inferred.

Declared optional metadata is copied to the plan. Supply output_tokens and/or
request_count only when those counts are known for the entire simulated
sequence; the corresponding rates use the full execution time in seconds.
Missing counts omit their rate columns. Zero duration with known counts leaves
the rate cells blank, without fabricating zero throughput.
"""
    # Labels become filenames; validate them before creating any output.
    for label in (model_name, *stages):
        if not label or label in (".", "..") or "/" in label or "\\" in label:
            raise ValueError(f"Expected a filename-safe model/stage name: {label!r}")
    reserved = {"stage_metrics", f"{model_name}_plan_metrics"}
    if reserved.intersection(stages):
        raise ValueError("Stage name conflicts with a summary CSV filename")
    directory = Path(output_dir)
    plan_path = directory / f"{model_name}_plan_metrics.csv"
    previous_files = set()
    if plan_path.exists():
        with plan_path.open(newline="", encoding="utf-8") as file:
            previous = next(csv.DictReader(file), {})
        try:
            manifest = json.loads(previous.get("operator_stage_csvs", "{}"))
        except (TypeError, ValueError):
            manifest = {}
        if isinstance(manifest, dict):
            previous_files = {filename for label, filename in manifest.items()
                              if isinstance(filename, str)
                              and filename == f"{label}.csv"
                              and Path(filename).name == filename
                              and "/" not in filename and "\\" not in filename}
    stage_rows = []
    csv_payloads = {}
    normalized = {
        name: value if isinstance(value, StageResult) else StageResult(
            [GraphResult(name, value)])
        for name, value in stages.items()
    }
    compute_dtypes, weight_dtypes = set(), set()
    for name, stage in normalized.items():
        rows, metrics = _semantic_stage(name, stage, model_name, hardware_name)
        fields = csv_fields("operator", model_name, rows)
        csv_payloads[f"{name}.csv"] = (fields, rows)
        stage_rows.append(metrics)
        for row in rows:
            if row.get("flops", 0) > 0:
                compute_dtypes.update(row["compute_dtype"].split("+"))
            weight_dtypes.update(filter(None, row.get("weight_storage_dtype", "").split("+")))
    csv_payloads["stage_metrics.csv"] = (
        csv_fields("stage", model_name, stage_rows), stage_rows)

    plan = project_plan_metadata(model_name, hardware_name, metadata)
    duration_ms = sum(row["execution_time_ms"] for row in stage_rows)
    fits = plan.get("fits_peak_stage_memory", True) and all(
        used <= memory.capacity_gb * 1e9
        for stage in normalized.values() for graph in stage.graphs
        for memory, used in graph.result.peak_memory.items())
    valid = fits and plan.get("is_valid", True)
    plan.update({
        "model_name": model_name,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "chip_name": hardware_name,
        "stages": ",".join(stages),
        "compute_dtype": plan.get("compute_dtype", "+".join(sorted(compute_dtypes))),
        "weight_storage_dtype": plan.get(
            "weight_storage_dtype", "+".join(sorted(weight_dtypes))),
        "peak_stage_memory_bytes": plan.get("peak_stage_memory_bytes", max(
            (row["peak_memory_bytes"] for row in stage_rows), default=0)),
        "fits_peak_stage_memory": fits,
        "execution_time_ms": duration_ms,
        "measured_execution_time_ms": sum(
            row["measured_execution_time_ms"] for row in stage_rows),
        "throughput": plan["output_tokens"] * 1000 / duration_ms
            if duration_ms > 0 and plan.get("output_tokens") is not None else "",
        "request_per_second": plan["request_count"] * 1000 / duration_ms
            if duration_ms > 0 and plan.get("request_count") is not None else "",
        "is_valid": valid,
        "fail_reason": "" if valid else plan.get("fail_reason", "Peak memory exceeds capacity"),
        "operator_stage_csvs": {name: f"{name}.csv" for name in stages},
    })
    csv_payloads[plan_path.name] = (csv_fields("plan", model_name, [plan]), [plan])
    summary = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "model_name": model_name, "hardware_name": hardware_name,
        "plan": plan, "stages": stage_rows,
    }
    json_payloads = {}
    if reproducibility is not None:
        json_payloads = {
            "summary.json": summary,
            "config.snapshot.json": reproducibility["configuration"],
            "provenance.json": reproducibility["provenance"],
        }
    # Validate all JSON before publishing any report, including the CSVs.
    json_payloads = {name: json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n"
                     for name, value in json_payloads.items()}

    # Prepare every report before replacing any published CSV. A failure while
    # constructing rows or serializing metadata leaves the previous run intact.
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".metrics-", dir=directory) as tmp:
        temporary = Path(tmp)
        for filename, (fields, rows) in csv_payloads.items():
            _write_csv(temporary / filename, fields, rows)
        for filename, value in json_payloads.items():
            (temporary / filename).write_text(value, encoding="utf-8")
        for filename in (*csv_payloads, *json_payloads):
            (temporary / filename).replace(directory / filename)
    # Remove only files owned by this model's previous successful export.
    # This keeps a reused output directory accurate when workflow stages change.
    current_files = {f"{name}.csv" for name in stages} | {plan_path.name, "stage_metrics.csv"}
    for filename in previous_files - current_files:
        (directory / filename).unlink(missing_ok=True)
    return summary
