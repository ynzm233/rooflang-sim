# 输出文件、单位与统计口径

运行后先看 `summary.json` 的 `plan.is_valid`、`fail_reason`、耗时和统计范围，再看 `stage_metrics.csv` 定位阶段，最后查看对应的算子 CSV。`is_valid=true` 只表示已建模约束通过，不能替代实机准确度或完整端到端覆盖的验证。

正式 CLI 与 Python `run_and_export()` 生成以下文件。低层 `export_metrics()` 只导出 CSV；`workflow.run()` 只返回内存中的结果。

| 文件 | 内容与用途 |
| --- | --- |
| `<stage>.csv` | 一个实际阶段的算子记录；多分支、多窗口和重复图共用此文件，用 `graph_label` 与 `graph_multiplier` 区分 |
| `stage_metrics.csv` | 每个已执行阶段一行：耗时、算量、流量、内存、调用次数和计时范围 |
| `<model>_plan_metrics.csv` | 本次运行的总体指标、生效部署信息、有效性、阶段清单和模型扩展统计 |
| `summary.json` | 同次导出的计划和阶段原始数值对象，适合程序处理；不包含逐算子时间线 |
| `config.snapshot.json` / `provenance.json` | 前者记录输入、生效配置和硬件数值；后者记录包/依赖版本、来源仓库与本地包哈希，用于复现和审计 |

例如默认 MAGI I2V 1080p 的阶段文件为 `text_encode.csv`、`image_vae_encode.csv`、`preview_denoise.csv`、`refiner_denoise.csv`、`video_vae_decode.csv`、`audio_vae_decode.csv`。T2V 不含图像编码；272p/540p 不含精修。MiniMax-H3 的阶段集合由其工作流声明。始终以本次 `plan.stages` 和 `operator_stage_csvs` 为准，不假设每个模型都有相同阶段。

## 1. CSV 公共约定

时间字段后缀 `_ms` 的单位是毫秒；内部仿真器使用微秒。流量和存储统一为字节，FLOPs 是浮点运算次数。硬件配置中的 GB/s、GB、TFLOP/s 使用十进制换算。形状、精度分项和复杂元数据在 CSV 单元格中使用 JSON 编码。

当前 CSV schema 版本为 `export_schema_version=2`。列定义集中于 [`src/rooflang/runtime/export_config.py`](../src/rooflang/runtime/export_config.py)。公共基础列保留；未定义的可选列可能从表头省略。空单元格表示未提供、不适用或未估计，需要结合 `timing_status` 等字段理解；数字 `0` 表示已定义的零值，不能互换。

### 算子 CSV

| 字段组 | 含义 |
| --- | --- |
| `model_name`, `hardware_name`, `node_name`, `operator`, `op_type`, `shape` | 来源模型/硬件、节点身份、算子类型和输入/权重/输出的形状；模型可以提供更具体的语义标签 |
| `execution_time_ms`, `start_ms`, `end_ms`, `device_ms` | 该核在图内的经过时间、起止时刻与本地资源执行时间；通信及资源竞争下这些量不能简单互换 |
| `compute_ms`, `memory_ms`, `network_ms`, `bound` | 理想计算、访存、网络分项及瓶颈标签；分项不是阶段墙钟时间的可加分解 |
| `flops`, `flops_by_dtype`, `compute_dtype`, `weight_storage_dtype`, `communicate_dtype` | 数学 FLOPs、用于计时的精度分项，以及计算/权重存储/通信精度。来源中的校准倍率可能使计时算量与数学 FLOPs 不同 |
| `graph_label`, `graph_multiplier`, `row_kind`, `device`, `timing_status`, `timing_included` | 图来源、重复次数、行类别、CPU/加速器类别及是否计时；重复图通常只写一份行，不能只数行数估算总工作量 |

`device_names` 记录设备名字，`stream` 是执行流。集合通信通常有多个参与者的 trace 条目，但导出为一个逻辑核记录，避免重复计数；不能再按设备名字数量乘一次 FLOPs 或流量。`start_ms/end_ms` 属于单次图内局部时间线，不是完整多阶段请求的绝对时间线。

| 字段 | 含义 |
| --- | --- |
| `activation`, `weight_read`, `memory_traffic_bytes` | 激活读写流量、此次读取的权重流量、两者之和；都不等于显存峰值 |
| `weight_saved`, `resident_weight_bytes` | 默认的权重存储量；模型提供来源口径时，前者保留来源的权重保存统计，后者另列关联后的物理权重大小。共享权重的复用行可有 `weight_saved=0` |
| `activation_peak_bytes_proxy` | 单核输入/输出或模型指定的激活近似；不是分配器测得的峰值，不宜逐行相加 |
| `overlapped_compute_ms`, `overlapped_memory_ms` | 理想 Roofline 的瓶颈归属：较大项保留、较小项记 0；相等时两项均保留。不能相加重构实际耗时；通信和未计时 CPU 行留空 |
| `kv_storage_dtype`, `kv_cache_read`, `kv_cache_saved`, `kv_cache_load` | 有对应建模事实时才出现的 KV 精度、读取、存储或加载统计；没有该统计的模型不补假零 |

`op_backend` 标识开销后端。`tag_mask`、`cpu_flops`、`cpu_flops_breakdown` 以及模型的 `component/module/layer_idx`、分支、子调用和来源索引等扩展列，仅在适用时出现。CPU 的 `cpu_flops_breakdown` 是 JSON 字典；未标定 CPU 延迟的行会显示 `timing_included=False`。

### 阶段 CSV

| 字段组 | 含义 |
| --- | --- |
| `execution_time_ms`, `measured_execution_time_ms`, `measurement_start_ms` | 完整阶段时间、测量窗口时间和测量起点统计；完整时间包含图内已建模的预加载，测量窗口可排除起点之前工作 |
| `compute_time_ms`, `memory_time_ms`, `comm_time_ms`, `total_flops`, `total_mem_bytes` | 对唯一逻辑核统计、再乘图调用次数的工作量分项；重叠时不能以资源时间之和替代经过时间 |
| `total_weight_bytes`, `peak_activation_bytes`, `peak_memory_bytes` | 图内权重按执行设备及共享身份去重；阶段默认取各图最大值。内存峰值是最大单个 memory 节点使用量，模型可用更具体的工作集估计覆盖 |
| `invocations`, `operator_count`, `device`, `timing_status` | 图调用次数及附加未计时条目调用数、实际导出行数、设备类别、计时状态；`operator_count` 不按 `graph_multiplier` 展开 |
| `cpu_total_flops`, `cpu_weight_bytes`, `cpu_peak_memory_bytes_proxy`, `cpu_execution_time_ms` | CPU 工作的独立统计。没有 CPU 工作或 CPU 延迟未估计时，耗时可为空；CPU 未计时行的 FLOPs 另记在 `cpu_total_flops`，不自动加入加速器的 `total_flops` |

多个图可能顺序执行，也可能共享权重。通用导出器不会猜测跨图/跨阶段的全部驻留关系；模型需要通过 `StageResult.metrics` 和工作集建模提供修正。不能把每阶段峰值或每阶段权重直接相加当作整机峰值。

### 计划 CSV

| 字段组 | 含义 |
| --- | --- |
| `model_name`, `chip_name`, `export_schema_version`, `stages` | 模型、硬件、schema 版本及本次实际阶段 |
| `batch_size`, `prompt_len`, `total_cards`, `main_cp/main_dp/main_ep/main_pp` | 模型提供的工作负载与并行度；未声明项为空，不推断成默认 1 |
| `execution_time_ms`, `measured_execution_time_ms` | 各已执行阶段时间之和；仅运行单阶段时并非完整请求耗时 |
| `peak_stage_memory_bytes`, `fits_peak_stage_memory`, `is_valid`, `fail_reason` | 阶段峰值及容量/模型约束判定；与 `memory_status`、`capacity_verdict_is_provisional` 等模型字段一起解释 |
| `throughput`, `request_per_second` | 仅在有明确 `output_tokens` / `request_count` 时出现，分别为 tokens/s 和 requests/s；分母使用本次完整已建模时间 |

MAGI 的 `request_per_second` 表示已建模加速器工作范围内的 clips/s，不能视为包含 CPU、冷加载、封装的完整服务吞吐。`end_to_end_timing_complete=False`、`timing_status`、`compute_scope` 和内存状态字段用于声明这些限制。其他模型也必须按自己的统计范围解释吞吐。

## 2. 重复图与结果重建

对于已计时工作，一次逻辑核记录的 FLOPs/流量只代表一次图调用。阶段总算量与流量为：

```text
阶段 FLOPs = Σ(算子 flops × graph_multiplier)
阶段流量  = Σ(算子 memory_traffic_bytes × graph_multiplier)
阶段时间  = Σ(各图的仿真经过时间 × 对应 multiplier)
计划时间  = Σ(已执行阶段时间)
```

只有图内完全串行时，阶段时间才等于各算子耗时乘次数的总和。并行/重叠/资源竞争应使用仿真器返回的经过时间。未计时行不参与时间求和；CPU 未计时算量按独立 CPU 字段解释。驻留权重和内存峰值不乘调用次数。

以 `python examples/toy_model.py --tokens 2 --hidden 4 --steps 3` 为例，`transform.csv` 只有一条核记录，`graph_multiplier=3`；阶段得到 48 FLOPs、144 字节访存，驻留权重仍为 16 字节。示例的 `--check` 会验证这一关系。

## 3. 三个 JSON 的结构

`summary.json` 顶层字段为 `schema_version`、`model_name`、`hardware_name`、`plan` 和 `stages`。`plan` 为计划对象，`stages` 为阶段对象列表；数值保留数值类型，字典/列表保留结构，不需要再解析 CSV 单元格。JSON 可以保留未投影到 CSV 列的统计字段；具体 CSV 列仍以导出配置为准。

`config.snapshot.json` 记录：

| 字段 | 含义 |
| --- | --- |
| `model_name`, `hardware_name`, `selected_stage` | 运行所选模型、硬件和阶段 |
| `input_config` | 本次 JSON 与显式 CLI 工作负载选项合并后的输入，尚未全部填充模型默认值 |
| `resolved_config` | 模型解析后的生效配置；MAGI 包含 dataclass 中的默认值和架构，其他模型可通过 metadata 的 `_resolved_config` 提供 |
| `hardware` | 节点与链路数值快照，包括精度吞吐、容量、带宽和延迟；Python API 未传入硬件对象时为 `null`，仍保留硬件名字 |

Python 接入时应调用 `run_and_export(..., config=config, hardware=hardware)`，并让模型提供完整 `_resolved_config` 或 `_model_config`。否则输入可能为空、生效配置可能只能回退到输入，无法充分检查默认值漂移。

`provenance.json` 顶层为 `package_version`、`python`、`dependencies`、`source_repositories` 和 `package_sha256`。当前依赖记录包含运行核心 `networkx`；不等于完整 Python 环境锁文件。`source_repositories` 保存迁移源仓库的版本记录；`package_sha256` 覆盖已安装 `rooflang` 包内 `.py/.json` 的相对路径及内容，包含默认配置。它不覆盖仓库所有文档、测试或输出目录，也不能代替硬件实测校准。

## 4. 用快照重跑同一输入

**不要把整个 `config.snapshot.json` 直接传给 `--config`。** CLI 接受模型工作负载 JSON，快照还包含运行标识、硬件和审计字段。应取出 `input_config`，同时复用原模型、硬件和阶段。

在相同版本的已安装仓库环境运行下面脚本。将第一行目录替换为需要重跑的输出目录，脚本会先核对包哈希，再输出到其 `replay/` 子目录：

```python
from pathlib import Path
import json
import subprocess
from rooflang.runtime.reproducibility import provenance

original = Path("outputs/magi2_preview")
snapshot = json.loads((original / "config.snapshot.json").read_text())
saved = json.loads((original / "provenance.json").read_text())
current = provenance()
for field in ("package_sha256", "python", "dependencies"):
    if saved[field] != current[field]:
        raise SystemExit(f"Environment differs: {field}; restore it before replay")
replay_input = original / "replay.input.json"
replay_input.write_text(json.dumps(snapshot["input_config"], indent=2) + "\n")
subprocess.run([
    "rooflang-sim", "--model", snapshot["model_name"],
    "--hardware", snapshot["hardware_name"],
    "--stage", snapshot["selected_stage"],
    "--config", str(replay_input), "--output-dir", str(original / "replay"),
], check=True)
```

此脚本适用于已注册模型和 CLI 硬件预设。自定义 Python 硬件或 `toy_affine` 应使用其原来的 Python 构建入口重跑；硬件快照本身不是自动硬件加载器。重跑后比较两次 `resolved_config`、`hardware` 和结果指标；外部测量或数值后端造成浮点差异时，使用明确的绝对/相对容差。

## 5. 目录复用与未覆盖工作

复用同一输出目录会覆盖本次同名 CSV 和 JSON。成功的 CSV 导出会依据上一份计划记录的 `operator_stage_csvs` 删除该模型已不再使用的阶段 CSV，保留无关文件；不应在一个目录混放不同模型或不同配置的正式基线。每次实验用独立输出目录最容易比较。

默认不会导出逐算子时间线 JSON；`summary.json` 是汇总。显式调试时可调用 `runtime.trace_export.export_trace`，但单个图的时间线仍需自行解释阶段位置和重复次数。

未知 CPU 时间、未建模工作区、真实分配器碎片、冷加载或其他遗漏，不会因为 CSV 存在就自动得到估计。使用报告时保留模型给出的范围与近似标签，并结合其独立复现说明判断适用性。
