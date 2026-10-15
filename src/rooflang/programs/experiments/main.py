# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.
"""Run a portable multimodal workflow and export its reproducible report."""

import argparse
import json
from pathlib import Path

from rooflang.programs.models import MODEL_NAMES, load_model
from rooflang.programs.experiments.run_and_export import run_and_export
from rooflang.programs.presets.b300 import B300Cluster, B300SuperChip
from rooflang.programs.presets.h200 import H200Cluster, H200SuperChip
from rooflang.runtime.stages import StageResult


def _a5():
    from rooflang.programs.models.minimax_h3.hardware import A5PRServer
    return A5PRServer(n_devices=4)


HARDWARE_MAP = {
    "B300Cluster1Node": lambda: B300Cluster(n_nodes=1),
    "B300SuperChip": B300SuperChip,
    "H200Cluster1Node": lambda: H200Cluster(n_nodes=1),
    "H200SuperChip": H200SuperChip,
    "A5_pr_server": _a5,
    "a5_pr_server": _a5,
}


def _invalid_constant(value):
    raise ValueError(f"Configuration must use finite JSON numbers, got {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Multimodal performance simulation (no model weights)")
    parser.add_argument("--model", choices=MODEL_NAMES, default="magi2_preview")
    parser.add_argument("--hardware", choices=list(HARDWARE_MAP),
                        help="Default: B300Cluster1Node for MAGI; A5_pr_server for MiniMax-H3")
    parser.add_argument("--config", type=Path, help="Flat model/workload configuration JSON")
    parser.add_argument("--stage", default="all", help="all, or one model-defined stage")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cp", type=int)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--workflow", help="MAGI: i2v/t2v; MiniMax-H3: fl2va/ref2va")
    parser.add_argument("--resolution", help="MAGI output resolution")
    parser.add_argument("--preview-steps", type=int)
    parser.add_argument("--refiner-steps", type=int)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"),
                            parse_constant=_invalid_constant) if args.config else {}
        if not isinstance(config, dict):
            raise ValueError("--config must contain a JSON object")
        for key in ("batch_size", "cp", "ep", "workflow", "resolution",
                    "preview_steps", "refiner_steps"):
            value = getattr(args, key)
            if value is not None:
                config[key] = value
        hardware_name = args.hardware or (
            "A5_pr_server" if args.model == "minimax_h3" else "B300Cluster1Node")
        hardware = HARDWARE_MAP[hardware_name]()
        workflow = load_model(args.model).create_workflow(hardware, config)
        results, output_dir = run_and_export(
            workflow, model_name=args.model, hardware_name=hardware_name,
            stage=args.stage, output_dir=args.output_dir, config=config, hardware=hardware)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for name, result in results.items():
        duration_ms = (sum(graph.result.measured_time_us * graph.multiplier
                           for graph in result.graphs) / 1000
                       if isinstance(result, StageResult) else result.measured_time_us / 1000)
        print(f"{name} ({hardware_name}): {duration_ms:.6f} ms")
    print(f"Reports: {output_dir}")


if __name__ == "__main__":
    main()
