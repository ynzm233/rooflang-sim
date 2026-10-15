# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""CSV export preserves scheduling, storage and workload accounting."""

import csv
import json
from fractions import Fraction
from types import SimpleNamespace

import pytest

from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.forward import Nop
from rooflang.language.kernels.comm import AllReduce
from rooflang.language.kernels.identity import Concat, Spawn
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.tensor import Tensor
from rooflang.runtime.metrics_export import export_metrics
from rooflang.runtime.simulator import Bound, SimulationResult, TraceEntry
from rooflang.runtime.stages import GraphResult, StageResult


class SyntheticKernel(Kernel):
    def __init__(self, flops=0.0, flops_by_dtype=None, **kwargs):
        super().__init__(**kwargs)
        self._flops = flops
        self._flops_by_dtype = flops_by_dtype

    @property
    def flops(self):
        return self._flops

    @property
    def flops_by_dtype(self):
        return self._flops_by_dtype


def _entry(kernel, device, *, start=0.0, end=1000.0,
           compute=0.0, memory=0.0, network=0.0):
    return TraceEntry(
        kernel=kernel, device=device, stream=0,
        start_us=start, end_us=end,
        bound=Bound.NETWORK if network else Bound.COMPUTE,
        compute_time_us=compute, memory_time_us=memory,
        network_time_us=network,
        local_elapsed_time_us=max(compute, memory),
        network_elapsed_time_us=network,
    )


def _rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_semantic_stage_filenames_follow_model_and_emit_only_csv(tmp_path):
    names = ["text_encode", "image_vae_encode", "preview_denoise",
             "refiner_denoise", "video_vae_decode", "audio_vae_decode"]
    stages = {name: StageResult([GraphResult(f"{name}_internal", SimulationResult([], 1000, {}))])
              for name in names}

    export_metrics(stages, tmp_path, model_name="magi2_preview")

    assert {path.name for path in tmp_path.iterdir()} == {
        *(f"{name}.csv" for name in names),
        "stage_metrics.csv", "magi2_preview_plan_metrics.csv",
    }
    assert [row["name"] for row in _rows(tmp_path / "stage_metrics.csv")] == names
    plan, = _rows(tmp_path / "magi2_preview_plan_metrics.csv")
    assert plan["stages"] == ",".join(names)
    assert json.loads(plan["operator_stage_csvs"]) == {name: f"{name}.csv" for name in names}
    assert float(plan["execution_time_ms"]) == 6


def test_graph_branches_and_windows_share_stage_and_reconcile_weighted_totals(tmp_path):
    gpu = Compute("gpu0", {"bf16": 1})
    hbm = Memory("hbm", capacity_gb=1)
    shared_weight = Tensor("bf16", (4, 4), weight_id="resident-model")

    def result(size, flops, duration, peak):
        kernel = SyntheticKernel(flops=flops,
            inputs={"x": Tensor("bf16", (size,))},
            outputs={"y": Tensor("bf16", (size,))},
            weights={"w": shared_weight})
        return SimulationResult(
            [_entry(kernel, gpu, end=duration, compute=duration / 2,
                    memory=duration / 4, network=duration / 8)],
            duration, {hbm: peak}, measurement_start_us=duration / 10)

    cond = result(4, 100, 1000, 100)
    uncond = result(8, 200, 2000, 150)
    stages = {
        "refiner_denoise": StageResult([
            GraphResult("refiner_cond", cond, 3),
            GraphResult("refiner_uncond", uncond, 3),
        ]),
        "video_vae_decode": StageResult([
            GraphResult("window_8", cond, 2),
            GraphResult("window_9", uncond),
        ]),
    }

    export_metrics(stages, tmp_path, model_name="video")

    operators = _rows(tmp_path / "refiner_denoise.csv")
    assert [row["graph_label"] for row in operators] == ["refiner_cond", "refiner_uncond"]
    assert [float(row["execution_time_ms"]) for row in operators] == [1, 2]
    assert [float(row["flops"]) for row in operators] == [100, 200]
    assert [int(row["graph_multiplier"]) for row in operators] == [3, 3]
    rows = {row["name"]: row for row in _rows(tmp_path / "stage_metrics.csv")}
    refiner, decoder = rows["refiner_denoise"], rows["video_vae_decode"]
    assert float(refiner["execution_time_ms"]) == 9
    assert float(refiner["measured_execution_time_ms"]) == pytest.approx(8.1)
    assert float(refiner["compute_time_ms"]) == 4.5
    assert float(refiner["memory_time_ms"]) == 2.25
    assert float(refiner["comm_time_ms"]) == 1.125
    assert float(refiner["total_flops"]) == 900
    assert float(refiner["total_mem_bytes"]) == 336
    assert float(refiner["total_weight_bytes"]) == 32
    assert float(refiner["peak_activation_bytes"]) == 32
    assert float(refiner["peak_memory_bytes"]) == 150
    assert int(refiner["invocations"]) == 6
    assert int(refiner["operator_count"]) == 2
    assert float(decoder["execution_time_ms"]) == 4
    assert float(decoder["total_flops"]) == 400
    assert float(decoder["total_weight_bytes"]) == 32
    assert int(decoder["invocations"]) == 3
    for name, stage in rows.items():
        detail = _rows(tmp_path / f"{name}.csv")
        assert float(stage["total_flops"]) == sum(
            float(row["flops"]) * int(row["graph_multiplier"]) for row in detail)
        assert float(stage["total_mem_bytes"]) == sum(
            float(row["memory_traffic_bytes"]) * int(row["graph_multiplier"]) for row in detail)
    plan, = _rows(tmp_path / "video_plan_metrics.csv")
    assert float(plan["execution_time_ms"]) == 13
    assert float(plan["peak_stage_memory_bytes"]) == 150
    assert {path.name for path in tmp_path.iterdir()} == {
        "refiner_denoise.csv", "video_vae_decode.csv", "stage_metrics.csv", "video_plan_metrics.csv",
    }


def test_untimed_cpu_work_remains_separate_from_modeled_gpu_time(tmp_path):
    gpu = Compute("gpu0", {"bf16": 1})
    kernel = SyntheticKernel(flops=12)
    broadcast = SimulationResult([_entry(kernel, gpu)], 1000, {})
    cpu_row = {"node_name": "qwen_cpu_encode", "device": "cpu",
               "cpu_flops": 500, "graph_multiplier": 2}
    stage = StageResult(
        graphs=[GraphResult("text_broadcast", broadcast)],
        untimed_rows=[cpu_row],
        metrics={"cpu_weight_bytes": 2000, "cpu_peak_memory_bytes_proxy": 2500},
    )

    export_metrics({"text_encode": stage}, tmp_path, model_name="multimodal")

    gpu_row, exported_cpu = _rows(tmp_path / "text_encode.csv")
    assert exported_cpu["execution_time_ms"] == ""
    assert exported_cpu["flops"] == ""
    assert exported_cpu["cpu_flops"] == "500"
    assert exported_cpu["graph_multiplier"] == "2"
    assert exported_cpu["timing_included"] == "False"
    assert exported_cpu["timing_status"] == "not_estimated"
    assert float(gpu_row["execution_time_ms"]) == 1
    metrics, = _rows(tmp_path / "stage_metrics.csv")
    assert float(metrics["execution_time_ms"]) == 1
    assert float(metrics["total_flops"]) == 12
    assert float(metrics["cpu_total_flops"]) == 1000
    assert float(metrics["cpu_weight_bytes"]) == 2000
    assert float(metrics["cpu_peak_memory_bytes_proxy"]) == 2500
    assert metrics["cpu_execution_time_ms"] == ""
    assert metrics["timing_status"] == "partial_unmodeled_work"
    assert metrics["device"] == "accelerator+cpu"
    assert int(metrics["invocations"]) == 3
    assert cpu_row == {"node_name": "qwen_cpu_encode", "device": "cpu",
                       "cpu_flops": 500, "graph_multiplier": 2}


@pytest.mark.parametrize("multiplier", [0, -1, 1.5, True])
def test_graph_invocation_multiplier_requires_positive_integer(multiplier):
    with pytest.raises(ValueError, match="positive integer"):
        GraphResult("invalid", SimulationResult([], 0, {}), multiplier)


def test_stage_wall_time_and_resident_weights_are_not_operator_sums(tmp_path):
    gpu = Compute("gpu0", tflops={"bf16": 1.0})
    first = SyntheticKernel(
        flops=1e6, inputs={"x": Tensor("bf16", (4,))},
        weights={"w": Tensor("bf16", (4, 4), weight_id="shared")},
        outputs={"y": Tensor("bf16", (4,))},
    )
    second = SyntheticKernel(
        flops=2e6, inputs={"x": Tensor("bf16", (4,))},
        weights={"w": Tensor("bf16", (4, 4), weight_id="shared")},
        outputs={"y": Tensor("bf16", (4,))},
    )
    first.weight_read_fraction = second.weight_read_fraction = Fraction(1, 4)
    result = SimulationResult(
        trace=[
            _entry(first, gpu, start=100, end=1600, compute=1000, memory=700),
            _entry(second, gpu, start=500, end=2000, compute=500, memory=1200),
        ],
        total_time_us=2000,
        peak_memory={Memory("hbm0"): 64, Memory("hbm1"): 256},
        measurement_start_us=100,
    )

    export_metrics({"decode": result}, tmp_path, model_name="toy")

    operators = _rows(tmp_path / "decode.csv")
    stage, = _rows(tmp_path / "stage_metrics.csv")
    plan, = _rows(tmp_path / "toy_plan_metrics.csv")
    assert len(operators) == 2
    assert sum(float(row["execution_time_ms"]) for row in operators) == 3.0
    assert float(stage["execution_time_ms"]) == 2.0
    assert float(stage["measured_execution_time_ms"]) == 1.9
    assert float(stage["compute_time_ms"]) == 1.5
    assert float(stage["memory_time_ms"]) == 1.9
    assert float(stage["total_flops"]) == 3e6
    assert float(stage["total_mem_bytes"]) == 48
    assert float(stage["total_weight_bytes"]) == 32
    assert float(stage["peak_activation_bytes"]) == 16
    assert float(stage["peak_memory_bytes"]) == 256
    assert [float(row["weight_read"]) for row in operators] == [8, 8]
    assert [float(row["weight_saved"]) for row in operators] == [32, 32]
    assert all(float(row["graph_multiplier"]) == 1 for row in operators)
    assert "throughput" not in plan
    assert "request_per_second" not in plan


@pytest.mark.parametrize("weight_id", [None, "shared"])
def test_resident_weights_deduplicate_per_device(tmp_path, weight_id):
    gpu0, gpu1 = Compute("gpu0"), Compute("gpu1")
    weight = Tensor("bf16", (4, 4), weight_id=weight_id)
    kernels = [SyntheticKernel(weights={"w": weight}) for _ in range(3)]
    result = SimulationResult(
        trace=[_entry(kernel, device) for kernel, device in
               zip(kernels, [gpu0, gpu0, gpu1])],
        total_time_us=1000, peak_memory={},
    )

    export_metrics({"forward": result}, tmp_path, model_name="toy")

    stage, = _rows(tmp_path / "stage_metrics.csv")
    assert float(stage["total_weight_bytes"]) == 64
    assert float(stage["total_mem_bytes"]) == 96


def test_collective_participant_traces_are_exported_once(tmp_path):
    collective = AllReduce(total_bytes=100, world=2, dtype="bf16")
    collective.inputs = {f"x{rank}": Tensor("bf16", (50,)) for rank in range(2)}
    collective.outputs = {f"y{rank}": Tensor("bf16", (50,)) for rank in range(2)}
    result = SimulationResult(
        trace=[_entry(collective, Compute(name), end=250, network=250)
               for name in ["gpu1", "gpu0"]],
        total_time_us=250, peak_memory={},
    )

    export_metrics({"prefill": result}, tmp_path, model_name="toy")

    operator, = _rows(tmp_path / "prefill.csv")
    stage, = _rows(tmp_path / "stage_metrics.csv")
    assert operator["operator"] == "AllReduce"
    assert set(operator["device_names"].split("+")) == {"gpu0", "gpu1"}
    assert float(operator["network_ms"]) == 0.25
    assert float(stage["comm_time_ms"]) == 0.25
    assert float(stage["total_flops"]) == 25
    assert float(stage["total_mem_bytes"]) == 200
    assert float(operator["activation_peak_bytes_proxy"]) == 300
    assert float(stage["peak_activation_bytes"]) == 300


@pytest.mark.parametrize("alias_factory", [Nop, lambda: Spawn(2), Concat],
                         ids=["nop", "spawn", "concat"])
def test_structural_aliases_do_not_inflate_activation_peak(tmp_path, alias_factory):
    gpu = Compute("gpu0", tflops={"bf16": 1.0})
    kernel = SyntheticKernel(
        flops=1e6,
        inputs={"x": Tensor("bf16", (4,))},
        outputs={"y": Tensor("bf16", (4,))},
    )
    alias = alias_factory()
    alias.inputs = {"x": Tensor("int32", (1_000_000,))}
    alias.outputs = {"y": Tensor("int32", (1_000_000,))}
    result = SimulationResult(
        [_entry(kernel, gpu), _entry(alias, gpu, start=1000, end=1000)],
        total_time_us=1000, peak_memory={},
    )

    export_metrics({"decode": result}, tmp_path, model_name="toy")

    operators = _rows(tmp_path / "decode.csv")
    stage, = _rows(tmp_path / "stage_metrics.csv")
    plan, = _rows(tmp_path / "toy_plan_metrics.csv")
    alias_row, = [row for row in operators if row["operator"] == type(alias).__name__]
    assert len(operators) == 2
    assert json.loads(alias_row["shape"])["inputs"]["x"]["shape"] == [1_000_000]
    assert float(alias_row["activation_peak_bytes_proxy"]) == 0
    assert float(alias_row["memory_traffic_bytes"]) == 0
    assert float(stage["peak_activation_bytes"]) == 16
    assert plan["compute_dtype"] == "bf16"


def test_fractional_shapes_mixed_precision_and_unicode_roundtrip(tmp_path):
    kernel = SyntheticKernel(
        flops=6e6, flops_by_dtype={"fp8": 2e6, "fp4": 4e6},
        inputs={"输入": Tensor("bf16", (Fraction(1, 2), Fraction(4, 1)))},
        outputs={"输出": Tensor("fp32", (2,))},
    )
    result = SimulationResult(
        trace=[_entry(kernel, Compute("gpu0", {"fp8": 2.0, "fp4": 4.0}))],
        total_time_us=1000, peak_memory={},
    )

    export_metrics({"prefill": result}, tmp_path, model_name="测试模型",
                   hardware_name="测试硬件",
                   metadata={"note": "引号\"、逗号,\n换行",
                             "compute_dtype": "bf16", "weight_storage_dtype": "fp8"})

    operator, = _rows(tmp_path / "prefill.csv")
    plan, = _rows(tmp_path / "测试模型_plan_metrics.csv")
    assert set(operator["compute_dtype"].split("+")) == {"fp8", "fp4"}
    assert operator["model_name"] == "测试模型"
    assert operator["hardware_name"] == "测试硬件"
    shape = json.loads(operator["shape"])
    assert shape["inputs"]["输入"] == {"shape": [0.5, 4], "dtype": "bf16"}
    assert shape["outputs"]["输出"] == {"shape": [2], "dtype": "fp32"}
    assert plan["note"] == "引号\"、逗号,\n换行"
    assert plan["compute_dtype"] == "bf16"
    assert plan["weight_storage_dtype"] == "fp8"


def test_plan_totals_and_throughput_include_preload(tmp_path):
    stages = {
        "prefill": SimulationResult([], 3000, {}, measurement_start_us=1000),
        "decode": SimulationResult([], 7000, {}, measurement_start_us=2000),
    }

    export_metrics(stages, tmp_path, model_name="toy",
                   metadata={"output_tokens": 14, "request_count": 2,
                             "batch_size": 2})

    stage_rows = _rows(tmp_path / "stage_metrics.csv")
    plan, = _rows(tmp_path / "toy_plan_metrics.csv")
    assert float(plan["execution_time_ms"]) == 10
    assert float(plan["execution_time_ms"]) == sum(
        float(row["execution_time_ms"]) for row in stage_rows)
    assert float(plan["measured_execution_time_ms"]) == 7
    assert float(plan["throughput"]) == 1400
    assert float(plan["request_per_second"]) == 200
    assert plan["batch_size"] == "2"


@pytest.mark.parametrize("metadata", [None, {"output_tokens": 2, "request_count": 1}])
def test_empty_stage_writes_headers_and_leaves_undefined_rates_blank(tmp_path, metadata):
    export_metrics({"empty": SimulationResult([], 0, {})}, tmp_path,
                   model_name="toy", metadata=metadata)

    assert _rows(tmp_path / "empty.csv") == []
    with (tmp_path / "empty.csv").open(encoding="utf-8", newline="") as file:
        assert "execution_time_ms" in next(csv.reader(file))
    stage, = _rows(tmp_path / "stage_metrics.csv")
    plan, = _rows(tmp_path / "toy_plan_metrics.csv")
    assert float(stage["execution_time_ms"]) == 0
    assert float(stage["total_flops"]) == 0
    assert float(stage["peak_memory_bytes"]) == 0
    if metadata is None:
        assert "throughput" not in plan
        assert "request_per_second" not in plan
    else:
        assert plan["throughput"] == ""
        assert plan["request_per_second"] == ""


@pytest.mark.parametrize(
    "model_name,stage_name,batch_size,prompt_len",
    [("dsv4_flash", "prefill", 2, 32), ("kimi_k3", "decode_step", 8, 256)],
)
def test_model_and_workload_metadata_drive_each_export(
        tmp_path, model_name, stage_name, batch_size, prompt_len):
    kernel = SyntheticKernel(
        flops=batch_size * prompt_len * 1000,
        inputs={"x": Tensor("bf16", (batch_size, prompt_len))},
    )
    result = SimulationResult(
        [_entry(kernel, Compute("gpu0"), end=1000 * batch_size)],
        total_time_us=1000 * batch_size, peak_memory={},
    )
    metadata = {
        "batch_size": batch_size, "prompt_len": prompt_len,
        "workload": {"stage": stage_name, "output_tokens": batch_size},
        "output_tokens": batch_size,
    }

    export_metrics({stage_name: result}, tmp_path, model_name=model_name,
                   hardware_name="configured_hardware", metadata=metadata)

    assert {path.name for path in tmp_path.iterdir()} == {
        f"{stage_name}.csv", "stage_metrics.csv", f"{model_name}_plan_metrics.csv"}
    operator, = _rows(tmp_path / f"{stage_name}.csv")
    plan, = _rows(tmp_path / f"{model_name}_plan_metrics.csv")
    assert operator["model_name"] == model_name
    assert operator["graph_label"] == stage_name
    assert float(operator["flops"]) == batch_size * prompt_len * 1000
    assert json.loads(operator["shape"])["inputs"]["x"]["shape"] == [
        batch_size, prompt_len]
    assert plan["model_name"] == model_name
    assert plan["chip_name"] == "configured_hardware"
    assert plan["stages"] == stage_name
    assert int(plan["batch_size"]) == batch_size
    assert int(plan["prompt_len"]) == prompt_len
    assert json.loads(plan["workload"]) == metadata["workload"]
    assert float(plan["execution_time_ms"]) == batch_size
    assert float(plan["throughput"]) == 1000
    assert isinstance(metadata["workload"], dict)


def test_registered_model_uses_shared_schema_without_other_model_fields(tmp_path):
    export_metrics(
        {"prefill": SimulationResult([], 1000, {})}, tmp_path,
        model_name="dsv4_flash",
        metadata={"output_tokens": 0, "request_count": None,
                  "dit_tokens": {"unexpected": object()}, "workflow": "i2v"},
    )

    plan, = _rows(tmp_path / "dsv4_flash_plan_metrics.csv")
    assert plan["export_schema_version"] == "2"
    assert plan["throughput"] == "0.0"
    assert plan["fail_reason"] == ""
    assert "request_per_second" not in plan
    assert "request_count" not in plan
    assert "dit_tokens" not in plan
    assert "workflow" not in plan


def test_magi_config_projection_is_central_and_private_input_is_not_exported(tmp_path):
    config = SimpleNamespace(
        condition_type="t2v", cfg=2, thw_list=[(85, 60, 106)],
        vae_downsampling_factor=(4, 16, 16),
        patch_downsampling_factor=(1, 1, 1), cp_size=8,
    )
    export_metrics(
        {"preview_denoise": SimulationResult([], 1000, {})}, tmp_path,
        model_name="magi2_preview", hardware_name="B300Cluster1Node",
        metadata={"_model_config": config,
                  "resolved_shape_summary": {"preview_padded_tokens": 4096},
                  "max_ttft_ms": "", "target": ""},
    )

    plan, = _rows(tmp_path / "magi2_preview_plan_metrics.csv")
    assert plan["dit_tokens"] == "4096"
    assert plan["condition_type"] == "t2v"
    assert plan["thw_list"] == "(85,60,106)"
    assert plan["vae_downsampling_factor"] == "(4,16,16)"
    assert plan["main_chip"] == "B300Cluster1Node"
    assert plan["main_cards"] == "8"
    assert "_model_config" not in plan
    assert "max_ttft_ms" not in plan
    assert "target" not in plan
    assert "throughput" not in plan


def test_invalid_metadata_does_not_publish_partial_csvs(tmp_path):
    previous = tmp_path / "prefill.csv"
    previous.write_text("previous report\n", encoding="utf-8")

    with pytest.raises(TypeError):
        export_metrics(
            {"prefill": SimulationResult([], 1000, {})}, tmp_path,
            model_name="toy", metadata={"bad": {"value": object()}},
        )

    assert previous.read_text(encoding="utf-8") == "previous report\n"
    assert {path.name for path in tmp_path.iterdir()} == {"prefill.csv"}


@pytest.mark.parametrize("model_name", [
    "dsv4_pro", "dsv4_flash", "glm_5_2_fp8", "kimi_k3", "magi2_preview",
])
def test_common_optional_statistics_follow_available_data_for_every_model(tmp_path, model_name):
    kernel = SyntheticKernel(flops=12)
    kernel.csv_metadata = {"tag_mask": 0, "kv_storage_dtype": "int32",
                           "kv_cache_read": 0, "kv_cache_saved": None}
    result = SimulationResult([_entry(kernel, Compute("gpu0"))], 1000, {})
    stage = StageResult(
        [GraphResult("forward", result)],
        untimed_rows=[{"device": "cpu", "cpu_flops": 30,
                       "cpu_flops_breakdown": {"attention_flops": 10, "mlp_flops": 20}}],
    )

    export_metrics({"forward": stage}, tmp_path, model_name=model_name)

    operator, cpu = _rows(tmp_path / "forward.csv")
    assert operator["tag_mask"] == "0"
    assert operator["kv_storage_dtype"] == "int32"
    assert operator["kv_cache_read"] == "0"
    assert "kv_cache_saved" not in operator
    assert "kv_cache_load" not in operator
    assert "cube_compute_ms" not in operator
    assert "vector_compute_ms" not in operator
    assert operator["cpu_flops_breakdown"] == ""
    assert sum(json.loads(cpu["cpu_flops_breakdown"]).values()) == 30


def test_source_weight_charge_preserves_physical_storage_and_cpu_peak(tmp_path):
    weight = Tensor("bf16", (4, 4), weight_id="shared")
    owner = SyntheticKernel(weights={"w": weight})
    reuse = SyntheticKernel(weights={"w": weight})
    owner.csv_metadata = {"source_weight_saved": 32}
    reuse.csv_metadata = {"source_weight_saved": 0}
    cpu = Compute("cpu0", kind="cpu")
    result = SimulationResult([_entry(owner, cpu), _entry(reuse, cpu)], 2000, {})

    export_metrics({"forward": result}, tmp_path, model_name="magi2_preview")

    rows = _rows(tmp_path / "forward.csv")
    assert [float(row["weight_saved"]) for row in rows] == [32, 0]
    assert [float(row["resident_weight_bytes"]) for row in rows] == [32, 32]
    stage, = _rows(tmp_path / "stage_metrics.csv")
    assert float(stage["total_weight_bytes"]) == 32
    assert float(stage["cpu_peak_memory_bytes_proxy"]) == 32
    assert float(stage["cpu_execution_time_ms"]) == 2


def test_cpu_timing_distinguishes_absent_work_from_measured_zero(tmp_path):
    kernel = SyntheticKernel()
    gpu = SimulationResult([_entry(kernel, Compute("gpu0"), end=0)], 0, {})
    cpu = SimulationResult([_entry(kernel, Compute("cpu0", kind="cpu"), end=0)], 0, {})

    export_metrics({"gpu_only": gpu, "zero_cpu": cpu}, tmp_path, model_name="toy")

    gpu_row, cpu_row = _rows(tmp_path / "stage_metrics.csv")
    assert gpu_row["cpu_execution_time_ms"] == ""
    assert float(cpu_row["cpu_execution_time_ms"]) == 0
