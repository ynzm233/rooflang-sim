"""Failed JSON serialization must not mix outputs from different runs."""

import pytest

from rooflang.programs.experiments.run_and_export import run_and_export
from rooflang.programs.experiments.workflow import Workflow
from rooflang.runtime.simulator import SimulationResult


def test_nonfinite_summary_preserves_previous_complete_report(tmp_path):
    def workflow(duration):
        return Workflow({"forward": lambda: SimulationResult([], duration, {})}, {})

    run_and_export(workflow(1000), model_name="toy", hardware_name="test",
                   output_dir=tmp_path, config={"revision": 1})
    previous = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="JSON|finite|range"):
        run_and_export(workflow(float("nan")), model_name="toy", hardware_name="test",
                       output_dir=tmp_path, config={"revision": 2})

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == previous
