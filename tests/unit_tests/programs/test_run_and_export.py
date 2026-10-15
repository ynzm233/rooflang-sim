"""The shared workflow entry point writes one report after a complete run."""

import csv
from pathlib import Path

import pytest

from rooflang.programs.experiments.run_and_export import run_and_export
from rooflang.programs.experiments.workflow import Workflow
from rooflang.runtime.simulator import SimulationResult


def _result():
    return SimulationResult([], 0, {})


def _plan(path):
    with path.open(newline="", encoding="utf-8") as file:
        return next(csv.DictReader(file))


def test_python_entrypoint_uses_cli_default_path_and_exports_invalid_result(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    workflow = Workflow(
        {"preview": _result},
        {"workflow": "i2v", "is_valid": False, "fail_reason": "capacity limit"},
    )

    results, output_dir = run_and_export(
        workflow, model_name="video", hardware_name="B300Cluster1Node")

    assert list(results) == ["preview"]
    assert output_dir == Path("outputs/video_i2v_B300Cluster1Node")
    assert {path.name for path in output_dir.iterdir()} == {
        "preview.csv", "stage_metrics.csv", "video_plan_metrics.csv",
        "summary.json", "config.snapshot.json", "provenance.json"}
    plan = _plan(output_dir / "video_plan_metrics.csv")
    assert plan["is_valid"] == "False"
    assert plan["fail_reason"] == "capacity limit"


def test_python_entrypoint_honors_selected_stage_and_output_override(tmp_path):
    workflow = Workflow({"prefill": _result, "decode": _result}, {})
    destination = tmp_path / "reports"

    results, output_dir = run_and_export(
        workflow, model_name="text", hardware_name="H200",
        stage="decode", output_dir=destination)

    assert list(results) == ["decode"]
    assert output_dir == destination
    assert {path.name for path in destination.iterdir()} == {
        "decode.csv", "stage_metrics.csv", "text_plan_metrics.csv",
        "summary.json", "config.snapshot.json", "provenance.json"}
    assert _plan(destination / "text_plan_metrics.csv")["stages"] == "decode"


def test_python_entrypoint_does_not_export_failed_workflow(tmp_path):
    executed = []

    def fail():
        executed.append("second")
        raise RuntimeError("simulation failed")

    workflow = Workflow(
        {"first": lambda: executed.append("first") or _result(), "second": fail},
        {},
    )
    destination = tmp_path / "reports"
    destination.mkdir()
    sentinel = destination / "notes.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="simulation failed"):
        run_and_export(workflow, model_name="text", hardware_name="H200",
                       output_dir=destination)

    assert executed == ["first", "second"]
    assert list(destination.iterdir()) == [sentinel]


def test_output_directory_cannot_mix_reports_from_different_models(tmp_path):
    first = Workflow({"stage": _result}, {})
    run_and_export(first, model_name="first", hardware_name="H200", output_dir=tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    second = Workflow({"stage": _result}, {})
    with pytest.raises(ValueError, match="another model"):
        run_and_export(second, model_name="second", hardware_name="H200", output_dir=tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_invalid_snapshot_keeps_all_previous_reports_consistent(tmp_path):
    workflow = Workflow({"stage": _result}, {})
    run_and_export(workflow, model_name="demo", hardware_name="H200", output_dir=tmp_path,
                   config={"seed": 1})
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="JSON compliant"):
        run_and_export(workflow, model_name="demo", hardware_name="H200", output_dir=tmp_path,
                       config={"seed": float("nan")})
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
