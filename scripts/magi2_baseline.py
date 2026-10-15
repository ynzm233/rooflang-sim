#!/usr/bin/env python3
"""Capture/verify MAGI-2 numerical regressions without external model assets.

Capture source baselines only after reviewing the source revision; normal use is
``python scripts/magi2_baseline.py`` to compare the installed package against them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from rooflang.programs.models.magi2_preview.workflow import create_workflow
from rooflang.programs.presets.b300 import B300Cluster
from rooflang.runtime.metrics_export import _semantic_stage


PROFILES = {
    "default": {},
    "minimal": {"resolution": "272p", "preview_steps": 1, "refiner_steps": 1},
}
BASELINES = Path(__file__).resolve().parents[1] / "baselines" / "magi2_preview"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def snapshot(workflow_name, profile="default"):
    workflow = create_workflow(B300Cluster(n_nodes=1), {
        "workflow": workflow_name, **PROFILES[profile],
    })
    results = workflow.run()
    config = workflow.metadata["_model_config"]
    stages = {}
    for name, stage in results.items():
        rows, metrics = _semantic_stage(name, stage, "magi2_preview", "B300Cluster1Node")
        # Hash every operator's structure/work, preserving its execution order.
        # Timings are compared separately with a floating-point tolerance.
        structural_fields = (
            "node_name", "operator", "shape", "graph_label", "graph_multiplier",
            "compute_dtype", "weight_storage_dtype", "communicate_dtype",
            "flops", "memory_traffic_bytes", "weight_read", "weight_saved",
            "activation_peak_bytes_proxy", "row_kind", "cpu_flops",
        )
        structural_rows = [{key: row.get(key) for key in structural_fields} for row in rows]
        timed = [row for row in rows if row.get("timing_included")]
        weighted = {
            "operator_invocations": sum(row["graph_multiplier"] for row in timed),
            "flops": sum(row["flops"] * row["graph_multiplier"] for row in timed),
            "memory_traffic_bytes": sum(row["memory_traffic_bytes"] * row["graph_multiplier"] for row in timed),
            "operator_elapsed_ms": sum(row["execution_time_ms"] * row["graph_multiplier"] for row in timed),
        }
        stages[name] = {
            "metrics": metrics,
            "weighted_operators": weighted,
            "operator_structure_sha256": hashlib.sha256(_canonical(structural_rows).encode()).hexdigest(),
            "graphs": [{
                "label": graph.label,
                "multiplier": graph.multiplier,
                "total_time_us": graph.result.total_time_us,
                "trace_entries": len(graph.result.trace),
                "peak_memory_bytes": {memory.name: value for memory, value in graph.result.peak_memory.items()},
            } for graph in stage.graphs],
        }
    metadata = {key: value for key, value in workflow.metadata.items() if not key.startswith("_")}
    captured = {
        "schema_version": 1,
        "profile": profile,
        "hardware": {"name": "B300Cluster", "n_nodes": 1},
        "config": {key: value for key, value in asdict(config).items() if not key.startswith("_")},
        "architecture_sha256": hashlib.sha256(_canonical(config.architecture).encode()).hexdigest(),
        "metadata": metadata,
        "stages": stages,
        "total_execution_time_ms": sum(stage["metrics"]["execution_time_ms"] for stage in stages.values()),
    }
    # JSON normalization also makes tuple-valued shapes consistent after reload.
    return json.loads(_canonical(captured))


def assert_matches(actual, expected, path="root"):
    """Compare every field; allow only normal floating-point accumulation noise."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"{path}: keys differ"
        for key, value in expected.items():
            assert_matches(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: lengths differ"
        for index, value in enumerate(expected):
            assert_matches(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), f"{path}: {actual} != {expected}"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="default")
    parser.add_argument("--workflow", choices=("i2v", "t2v", "both"), default="both")
    parser.add_argument("--capture-dir", type=Path, help="Write newly captured snapshots instead of verifying")
    args = parser.parse_args()
    workflows = ("i2v", "t2v") if args.workflow == "both" else (args.workflow,)
    for name in workflows:
        actual = snapshot(name, args.profile)
        filename = f"{name}_{args.profile}.json"
        if args.capture_dir:
            args.capture_dir.mkdir(parents=True, exist_ok=True)
            path = args.capture_dir / filename
            path.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"captured {path}", flush=True)
        else:
            expected = json.loads((BASELINES / filename).read_text(encoding="utf-8"))
            assert_matches(actual, expected)
            print(f"PASS {filename}: {actual['total_execution_time_ms']:.9f} ms", flush=True)


if __name__ == "__main__":
    main()
