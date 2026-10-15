# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Run a model workflow and export its completed simulation results."""

from pathlib import Path

from rooflang.programs.experiments.workflow import Workflow
from rooflang.runtime.metrics_export import export_metrics
from rooflang.runtime.simulator import SimulationResult
from rooflang.runtime.stages import StageResult
from rooflang.runtime.reproducibility import config_snapshot, provenance


def run_and_export(
    workflow: Workflow, *, model_name: str, hardware_name: str,
    stage: str = "all", output_dir: str | Path | None = None,
    config: dict | None = None, hardware=None,
) -> tuple[dict[str, SimulationResult | StageResult], Path]:
    """Run selected semantic stages, then export their shared CSV report.

    A workflow failure propagates before export begins. Completed results are
    exported even when the workflow marks the plan invalid in its metadata.
    """
    label = workflow.metadata.get("workflow", stage)
    directory = (Path(output_dir) if output_dir is not None else
                 Path("outputs") / f"{model_name}_{label}_{hardware_name}")
    other_plans = [path for path in directory.glob("*_plan_metrics.csv")
                   if path.name != f"{model_name}_plan_metrics.csv"]
    if other_plans:
        raise ValueError("Output directory contains another model's report; choose a separate --output-dir")
    snapshot = config_snapshot(workflow, model_name=model_name,
                               hardware_name=hardware_name, stage=stage,
                               config=config, hardware=hardware)
    results = workflow.run(stage)
    export_metrics(results, directory, model_name=model_name,
                   hardware_name=hardware_name, metadata=workflow.metadata,
                   reproducibility={"configuration": snapshot, "provenance": provenance()})
    return results, directory
