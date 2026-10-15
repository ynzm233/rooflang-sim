# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Shared simulation and optional CSV export for configured models."""

from rooflang.runtime.simulator import Simulator
from rooflang.runtime.metrics_export import export_metrics


def simulate(
    g, p, hw, measurement_start=None, *,
    model_name: str | None = None, stage_name: str | None = None,
    output_dir=None, metadata=None, hardware_name: str | None = None,
):
    """Run DES and optionally export configured CSV metrics.

    Set output_dir, model_name and stage_name to write the operator, stage and
    plan CSVs for this invocation. Metadata carries the effective simulation
    configuration into the plan. Without output_dir, no files are written.
    The operator trace remains available in the returned simulation result.
    Use export_metrics with multiple results for a combined multi-stage plan.
    """
    if output_dir is not None and (not model_name or not stage_name):
        raise ValueError("CSV export requires model_name and stage_name")
    result = Simulator(
        g, p, hw, measurement_start=measurement_start).run()
    if output_dir is not None:
        export_metrics(
            {stage_name: result}, output_dir, model_name=model_name,
            hardware_name=hardware_name or type(hw).__name__, metadata=metadata)
    return result
