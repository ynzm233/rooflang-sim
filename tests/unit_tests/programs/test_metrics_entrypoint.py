"""CLI integration preserves model config and reproducible reports."""
import csv
import json
from types import SimpleNamespace
import pytest
from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.forward import Nop
from rooflang.programs.experiments import main as main_module
from rooflang.programs.experiments.workflow import Workflow
from rooflang.programs.presets.b300 import B300SuperChip
from rooflang.runtime.simulator import Bound, SimulationResult, TraceEntry
from rooflang.runtime.stages import GraphResult, StageResult


def _rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


@pytest.mark.parametrize("selected", [None, "image_vae_encode"])
def test_cli_delegates_semantic_stages_and_preserves_model_configuration(
    monkeypatch, tmp_path, selected,
):
    names = ["text_encode", "image_vae_encode", "preview_denoise",
             "refiner_denoise", "video_vae_decode", "audio_vae_decode"]
    loaded, executed = [], []
    config = {"workflow": "i2v", "resolution": "540p", "preview_steps": 3,
              "refiner_steps": 2, "batch_size": 4, "custom_input": {"tokens": 91}}
    config_path = tmp_path / "workload.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    hardware = B300SuperChip()
    gpu = Compute("custom_gpu", {"bf16": 1})

    def create_workflow(actual_hardware, effective_config):
        assert actual_hardware is hardware
        loaded.append(effective_config)

        def run(name):
            executed.append(name)
            trace = [TraceEntry(Nop(), gpu, 0, 0, 1000, Bound.COMPUTE,
                                1000, 0, 0, 1000, 0)]
            return StageResult([GraphResult(f"{name}_graph", SimulationResult(trace, 1000, {}))])

        return Workflow({name: lambda name=name: run(name) for name in names},
                        metadata=effective_config.copy())

    monkeypatch.setattr(main_module, "MODEL_NAMES", ("custom_multimodal",))
    monkeypatch.setattr(main_module, "load_model", lambda name: SimpleNamespace(create_workflow=create_workflow))
    monkeypatch.setattr(main_module, "HARDWARE_MAP", {"CustomChip": lambda: hardware})
    output_dir = tmp_path / "csv"
    arguments = ["rooflang", "--model", "custom_multimodal", "--hardware", "CustomChip",
                 "--config", str(config_path), "--preview-steps", "7", "--output-dir", str(output_dir)]
    if selected is not None:
        arguments += ["--stage", selected]
    monkeypatch.setattr("sys.argv", arguments)

    main_module.main()

    expected_names = names if selected is None else [selected]
    assert loaded == [{**config, "preview_steps": 7}]
    assert executed == expected_names
    assert json.loads(config_path.read_text()) == config
    assert {path.name for path in output_dir.iterdir()} == {
        *(f"{name}.csv" for name in expected_names),
        "stage_metrics.csv", "custom_multimodal_plan_metrics.csv",
        "summary.json", "config.snapshot.json", "provenance.json",
    }
    assert [row["name"] for row in _rows(output_dir / "stage_metrics.csv")] == expected_names
    plan, = _rows(output_dir / "custom_multimodal_plan_metrics.csv")
    assert plan["stages"] == ",".join(expected_names)
    assert plan["preview_steps"] == "7"
    assert plan["refiner_steps"] == "2"
    assert plan["resolution"] == "540p"
    assert json.loads(plan["custom_input"]) == {"tokens": 91}

    snapshot = json.loads((output_dir / "config.snapshot.json").read_text())
    assert snapshot["input_config"] == {**config, "preview_steps": 7}
    assert snapshot["hardware"]["nodes"]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert [row["name"] for row in summary["stages"]] == expected_names
    assert summary["plan"]["execution_time_ms"] == len(expected_names)
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert len(provenance["package_sha256"]) == 64
