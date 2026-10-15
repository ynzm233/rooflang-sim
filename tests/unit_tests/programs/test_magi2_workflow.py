"""Native MAGI-2 workflows export every selected semantic stage offline."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from rooflang.programs.models.magi2_preview import create_workflow
from rooflang.programs.presets.b300 import B300Cluster
from rooflang.runtime.metrics_export import export_metrics


def _rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


@pytest.mark.parametrize("workflow_name", ["i2v", "t2v"])
@pytest.mark.parametrize("resolution", ["1080p", "540p"])
def test_native_workflow_exports_actual_semantic_stages_and_reconciles_csv(
    tmp_path, workflow_name, resolution,
):
    workflow = create_workflow(B300Cluster(n_nodes=1), {
        "workflow": workflow_name, "resolution": resolution,
        "preview_steps": 3, "refiner_steps": 2,
    })
    expected = ["text_encode"]
    if workflow_name == "i2v":
        expected += ["image_vae_encode"]
    expected += ["preview_denoise"]
    if resolution == "1080p":
        expected += ["refiner_denoise"]
    expected += ["video_vae_decode", "audio_vae_decode"]

    results = workflow.run()
    assert list(results) == expected
    export_metrics(results, tmp_path, model_name="magi2_preview",
                   hardware_name="B300Cluster1Node", metadata=workflow.metadata)

    assert {path.name for path in tmp_path.iterdir()} == {
        *(f"{stage}.csv" for stage in expected),
        "stage_metrics.csv", "magi2_preview_plan_metrics.csv",
    }
    summaries = _rows(tmp_path / "stage_metrics.csv")
    assert [row["name"] for row in summaries] == expected
    for summary in summaries:
        rows = _rows(tmp_path / f"{summary['name']}.csv")
        assert rows
        timed = [row for row in rows if row["timing_included"] == "True"]
        duration = sum(float(row["execution_time_ms"]) * int(row["graph_multiplier"])
                       for row in timed)
        flops = sum(float(row["flops"]) * int(row["graph_multiplier"]) for row in timed)
        assert float(summary["execution_time_ms"]) == pytest.approx(duration)
        assert float(summary["total_flops"]) == pytest.approx(flops)
        assert int(summary["operator_count"]) == len(rows)
    plan, = _rows(tmp_path / "magi2_preview_plan_metrics.csv")
    applicable_source_fields = (
        "model_name dit_tokens prompt_len condition_type cfg num_diffusion_steps "
        "thw_list vae_downsampling_factor patch_downsampling_factor compute_dtype "
        "weight_storage_dtype kv_storage_dtype chip_name total_cards "
        "deploy_strategy stages main_chip main_cards main_concurrency "
        "main_dp main_tp main_pp main_ep main_sp main_cp main_moe_dp main_moe_tp "
        "main_mlp_tp main_oproj_tp workflow preview_forward_count "
        "refiner_forward_count peak_stage_memory_bytes all_resident_memory_bytes "
        "fits_peak_stage_memory fits_all_resident_memory resolved_shape_summary "
        "execution_time_ms request_per_second is_valid fail_reason"
    ).split()
    assert len(applicable_source_fields) == 41
    assert set(applicable_source_fields) <= plan.keys()
    assert plan["export_schema_version"] == "2"
    assert {"max_ttft_ms", "target", "throughput"}.isdisjoint(plan)
    shape = json.loads(plan["resolved_shape_summary"])
    assert plan["dit_tokens"] == str(shape["preview_padded_tokens"])
    assert plan["condition_type"] == workflow_name
    assert plan["cfg"] == "2"
    assert plan["thw_list"] == "(" + ",".join(map(str, shape["decoded_video_shape"][:3])) + ")"
    assert plan["vae_downsampling_factor"] == "(4,16,16)"
    assert plan["patch_downsampling_factor"] == "(1,1,1)"
    assert plan["kv_storage_dtype"] == "bf16"
    assert plan["main_chip"] == "B300Cluster1Node"
    assert plan["main_cards"] == plan["main_cp"] == plan["main_ep"] == "8"
    assert all(plan[field] == "1" for field in (
        "main_concurrency", "main_tp", "main_sp", "main_moe_dp",
        "main_moe_tp", "main_mlp_tp", "main_oproj_tp"))
    assert plan["deploy_strategy"] == "magi2_preview"
    assert float(plan["execution_time_ms"]) == pytest.approx(
        sum(float(row["execution_time_ms"]) for row in summaries))
    assert plan["workflow"] == workflow_name
    assert plan["resolution"] == resolution
    assert plan["preview_forward_count"] == "3"
    assert plan["refiner_forward_count"] == ("4" if resolution == "1080p" else "0")
    assert plan["end_to_end_timing_complete"] == "False"
    assert plan["timing_backend"] == "rooflang_discrete_event_simulator"
    assert json.loads(plan["operator_stage_csvs"]) == {name: f"{name}.csv" for name in expected}

    preview = results["preview_denoise"].graphs
    assert [(graph.label, graph.multiplier) for graph in preview] == [("preview_denoise", 3)]
    if resolution == "1080p":
        assert [(graph.label, graph.multiplier) for graph in results["refiner_denoise"].graphs] == [
            ("latent_upsample", 1), ("refiner_cond", 2), ("refiner_uncond", 2),
        ]
    windows = results["video_vae_decode"].graphs
    assert len(windows) == (9 if resolution == "1080p" else 5)
    assert [graph.label for graph in windows] == [
        f"video_decode_window_{index:02d}" for index in range(len(windows))]
    assert all(graph.multiplier == 1 for graph in windows)
    text, = [row for row in summaries if row["name"] == "text_encode"]
    assert text["cpu_execution_time_ms"] == ""
    assert text["timing_status"] == "partial_cpu_untimed"
    assert float(text["cpu_total_flops"]) > 0
    assert float(text["cpu_weight_bytes"]) == 51249200128
    cpu_rows = [row for row in _rows(tmp_path / "text_encode.csv") if row["device"] == "cpu"]
    assert [row["graph_label"] for row in cpu_rows] == ["qwen_positive", "qwen_negative"]
    assert all(row["execution_time_ms"] == "" and row["timing_included"] == "False" for row in cpu_rows)
    assert float(text["cpu_total_flops"]) == pytest.approx(
        sum(float(row["cpu_flops"]) for row in cpu_rows))
    for row in cpu_rows:
        breakdown = json.loads(row["cpu_flops_breakdown"])
        assert set(breakdown) == {
            "ffn_matmul_flops", "linear_attention_projection_flops",
            "full_attention_projection_flops", "causal_attention_matmul_flops",
            "depthwise_conv_flops", "gdn_chunk_matmul_flops",
            "gdn_triangular_recurrence_scalar_flops",
        }
        assert sum(breakdown.values()) == float(row["cpu_flops"])
        assert row["tag_mask"] == row["kv_storage_dtype"] == ""


def test_diffusion_steps_scale_invocations_without_changing_graph_or_resident_weights(tmp_path):
    stages = []
    for steps in (1, 4):
        workflow = create_workflow(B300Cluster(n_nodes=1), {
            "workflow": "t2v", "resolution": "540p", "preview_steps": steps,
        })
        result = workflow.run("preview_denoise")
        output = tmp_path / str(steps)
        export_metrics(result, output, model_name="magi2_preview", metadata=workflow.metadata)
        summary, = _rows(output / "stage_metrics.csv")
        detail = _rows(output / "preview_denoise.csv")
        stages.append((summary, detail))
    first, repeated = stages
    assert int(first[0]["invocations"]) == 1
    assert int(repeated[0]["invocations"]) == 4
    assert float(repeated[0]["execution_time_ms"]) == pytest.approx(4 * float(first[0]["execution_time_ms"]))
    assert float(repeated[0]["total_flops"]) == pytest.approx(4 * float(first[0]["total_flops"]))
    assert repeated[0]["total_weight_bytes"] == first[0]["total_weight_bytes"]
    assert repeated[0]["peak_memory_bytes"] == first[0]["peak_memory_bytes"]
    assert len(first[1]) == len(repeated[1])
    assert all(int(row["graph_multiplier"]) == 4 for row in repeated[1])
    assert [row["execution_time_ms"] for row in repeated[1]] == [row["execution_time_ms"] for row in first[1]]


def test_reused_output_directory_removes_only_previous_model_stage_files(tmp_path):
    hardware = B300Cluster(n_nodes=1)
    for workflow_name, resolution in [("i2v", "1080p"), ("t2v", "540p")]:
        workflow = create_workflow(hardware, {
            "workflow": workflow_name, "resolution": resolution,
            "preview_steps": 1, "refiner_steps": 1,
        })
        results = workflow.run()
        export_metrics(results, tmp_path, model_name="magi2_preview", metadata=workflow.metadata)
        if workflow_name == "i2v":
            assert (tmp_path / "image_vae_encode.csv").is_file()
            assert (tmp_path / "refiner_denoise.csv").is_file()
            (tmp_path / "notes.csv").write_text("note\nkeep this user file\n", encoding="utf-8")

    assert {path.name for path in tmp_path.iterdir()} == {
        "text_encode.csv", "preview_denoise.csv", "video_vae_decode.csv",
        "audio_vae_decode.csv", "stage_metrics.csv", "magi2_preview_plan_metrics.csv", "notes.csv",
    }
    assert (tmp_path / "notes.csv").read_text(encoding="utf-8") == "note\nkeep this user file\n"
    plan, = _rows(tmp_path / "magi2_preview_plan_metrics.csv")
    assert plan["workflow"] == "t2v"
    assert plan["resolution"] == "540p"
    assert plan["stages"] == "text_encode,preview_denoise,video_vae_decode,audio_vae_decode"


def test_cli_runs_with_network_and_external_file_reads_blocked(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    script = textwrap.dedent("""
        import json
        import os
        from pathlib import Path
        import sys

        repository, output = map(Path, sys.argv[1:3])
        sys.path.insert(0, str(repository / 'src'))
        roots = [repository.resolve(), output.resolve(),
                 Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
        input_json = set()

        def audit(event, arguments):
            if event.startswith('socket.'):
                raise AssertionError(f'Network access forbidden: {event}')
            if event != 'open' or isinstance(arguments[0], int):
                return
            path = Path(os.fsdecode(arguments[0])).resolve()
            if not any(path == root or root in path.parents for root in roots):
                raise AssertionError(f'External file access forbidden: {path}')
            if path.suffix == '.json' and path.name in ('architecture.json', 'workload.json'):
                input_json.add(str(path.relative_to(repository)))

        sys.addaudithook(audit)
        from rooflang.programs.experiments.main import main
        sys.argv = ['rooflang', '--model', 'magi2_preview', '--hardware', 'B300Cluster1Node',
                    '--workflow', 't2v', '--resolution', '540p', '--preview-steps', '2',
                    '--output-dir', str(output)]
        main()
        assert {
            'src/rooflang/programs/models/magi2_preview/architecture.json',
            'src/rooflang/programs/models/magi2_preview/workload.json',
        } <= input_json, input_json
        # Provenance hashing may inspect other packaged models' JSON files.
        assert all(path.startswith('src/rooflang/') for path in input_json), input_json
        print('OFFLINE_INPUTS_VERIFIED')
    """)
    output = tmp_path / "csv"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(repository), str(output)],
        cwd=repository, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "OFFLINE_INPUTS_VERIFIED" in completed.stdout
    assert {path.name for path in output.iterdir()} == {
        "text_encode.csv", "preview_denoise.csv", "video_vae_decode.csv",
        "audio_vae_decode.csv", "stage_metrics.csv", "magi2_preview_plan_metrics.csv",
        "summary.json", "config.snapshot.json", "provenance.json",
    }
