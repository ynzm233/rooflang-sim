"""Compare the independent source-run snapshots, including default 1080p jobs."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("magi2_baseline", ROOT / "scripts" / "magi2_baseline.py")
BASELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINE)


@pytest.mark.parametrize("workflow", ["i2v", "t2v"])
@pytest.mark.parametrize("profile", ["default", "minimal"])
def test_magi2_matches_frozen_source_run(workflow, profile):
    expected = json.loads((BASELINE.BASELINES / f"{workflow}_{profile}.json").read_text(encoding="utf-8"))
    actual = BASELINE.snapshot(workflow, profile)
    BASELINE.assert_matches(actual, expected)

    for stage in actual["stages"].values():
        metrics = stage["metrics"]
        weighted = stage["weighted_operators"]
        assert metrics["total_flops"] == pytest.approx(weighted["flops"], rel=1e-10)
        assert metrics["total_mem_bytes"] == pytest.approx(weighted["memory_traffic_bytes"], rel=1e-10)
        assert metrics["execution_time_ms"] == pytest.approx(sum(
            graph["total_time_us"] * graph["multiplier"] / 1000 for graph in stage["graphs"]
        ), rel=1e-10)
    assert actual["metadata"]["end_to_end_timing_complete"] is False
    assert actual["metadata"]["capacity_verdict_is_provisional"] is True
