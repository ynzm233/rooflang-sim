"""Model-defined workflow stages determine execution and aggregation."""

import pytest

from rooflang.programs.experiments.workflow import Workflow
from rooflang.runtime.simulator import SimulationResult
from rooflang.runtime.stages import StageResult


@pytest.mark.parametrize("names", [
    ["text_encode", "preview_denoise", "video_vae_decode", "audio_vae_decode"],
    ["image_vae_encode", "refiner_denoise"],
    ["custom_stage"],
])
def test_default_all_runs_exact_model_stages_in_order(names):
    calls = []
    expected = {name: StageResult() for name in names}

    def run(name):
        calls.append(name)
        return expected[name]

    workflow = Workflow({name: lambda name=name: run(name) for name in names}, metadata={})
    results = workflow.run()
    assert calls == names
    assert list(results) == names
    assert all(results[name] is expected[name] for name in names)


def test_explicit_stage_only_runs_selected_callback_and_summarizes_results():
    calls, summaries = [], []
    result = SimulationResult([], 25, {})

    def run(name):
        calls.append(name)
        return result

    def summarize(results):
        summaries.append(results)
        return {"executed_stages": list(results), "execution_time_us": 25}

    workflow = Workflow(
        {name: lambda name=name: run(name) for name in ["encode", "denoise", "decode"]},
        metadata={"model_name": "custom", "batch_size": 1}, summarize=summarize,
    )
    results = workflow.run("denoise")
    assert calls == ["denoise"]
    assert results == {"denoise": result}
    assert summaries == [results]
    assert workflow.metadata == {
        "model_name": "custom", "batch_size": 1,
        "executed_stages": ["denoise"], "execution_time_us": 25,
    }


def test_unknown_explicit_stage_reports_available_names_without_running_callbacks():
    calls = []
    workflow = Workflow({"text_encode": lambda: calls.append("executed")}, metadata={})
    with pytest.raises(ValueError, match="Unknown stage 'prefill'; available: text_encode"):
        workflow.run("prefill")
    assert calls == []


def test_empty_workflow_is_rejected():
    with pytest.raises(ValueError, match="Unknown stage 'all'"):
        Workflow({}, metadata={}).run()
