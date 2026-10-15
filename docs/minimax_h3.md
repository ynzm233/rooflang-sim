# MiniMax-H3：FL2VA / Ref2VA 性能仿真

## 运行与复现

在仓库根目录安装 `pip install -e '.[test]'` 后：

```bash
rooflang-sim --model minimax_h3 --hardware A5_pr_server \
  --config examples/minimax_h3_fl2va.json --output-dir outputs/h3-fl2va
rooflang-sim --model minimax_h3 --hardware A5_pr_server \
  --config examples/minimax_h3_ref2va.json --output-dir outputs/h3-ref2va
python scripts/minimax_h3_baseline.py
python -m pytest -q tests/test_minimax_h3.py
```

`--config` 是扁平模型参数 JSON；模型名、硬件名和输出目录通过 CLI 传入。省略 H3 硬件参数时默认 `A5_pr_server`。该模型输出性能和容量估算，不生成图片、视频、音频或模型权重。

两条默认流程均为 batch=1、SP=4、TP/CP/DP/PP=1、FP16、124 帧、768×1344、50 次去噪。源结果由 `model_Sim` 提交 `f27d4bdfa823fe0b151749cc5d1e1d734e7fb583` 的当前配置重新执行生成：

| 流程 | 含义 | packed rows | 源主算子时延 ms |
|---|---|---:|---:|
| FL2VA | 首帧、尾帧或首尾两帧约束生成带音轨视频；默认首尾两帧 | 41763 | 207504.38554284864 |
| Ref2VA | 参考图像/视频和可选参考音频约束生成带音轨视频；默认参考视频带音轨 | 81532 | 624059.5473882345 |

`baselines/minimax_h3/` 保存原始 summary、阶段 CSV、逐图算子 CSV、源 YAML 和来源 SHA。校验脚本逐行比对算子身份、层号、FLOPs、访存、权重和时间分项，并核对阶段次数、总量、shape、驻留容量。浮点容差为相对 `1e-10`、绝对 `1e-8`；不要求不同 Python 版本产生逐字节相同的浮点字符串。

如需从仍可访问的源库重新取证，用新输出目录执行；该命令不修改源配置，也不在源目录写字节码：

```bash
python scripts/minimax_h3_baseline.py \
  --source /path/to/model_Sim \
  --source-python /path/to/model_Sim/.venv/bin/python \
  --output /tmp/h3-fresh-source
```

普通使用和冻结基线校验均不需要两个旧仓库；仅 `--source` 的取证操作需要显式提供旧库与安装其依赖的解释器。

## 接入数据如何准备

输入是结构和形状数据，仿真不加载媒体或权重。`architecture.json` 定义组件、层数、通道、head、FFN、patch 和权重字节数；`workload.json` 给出完整默认请求，两份 examples 是可直接编辑的最小运行配置。

1. 准备目标 `target.frames/height/width`。宽高为 32 的正整数倍；目标帧数满足 `17*n+5`，并处于 24fps 下 5–15 秒范围。不要将 latent 尺寸写进这些像素/帧字段。
2. 准备条件。FL2VA 用 `keyframes: ["first"]`、`["last"]` 或两者；Ref2VA 用 `references`，每条含 `kind`（image/video/audio）、视觉尺寸和帧数、音频秒数。参考数量、帧数和持续时间由配置类校验；修改条件时同步下一项。
3. 从真实 Qwen processor/tokenizer 获取完整 `text_condition_rows`：包括 prompt、标签/时间戳、视觉占位、起止标记。不能只填自然语言 prompt 长度。仿真不实现 tokenizer。默认 prompt `Generate a cinematic video.` 为源记录的 5 tokens；FL2VA 为 `5 + 2×1016 = 2037`，Ref2VA 为 `6×1010 + 47 + 5 = 6112`。原始注释保存在 `baselines/minimax_h3/source/configs/workload/workload_minimax_h3.yaml`。
4. 准备运行参数 `num_inference_steps`、`sp`、`assume_stage_unload`。当前可选 SP=1/2/4/8，必须有相应设备，默认硬件含 4 卡；其余并行维度和 batch 固定为 1。其他计算精度明确报错，避免误把源 FP16 基线解释成混合精度结果。
5. 对新结构记录证据来源和缺失部分。当前 `indexed_weight_bytes` 原样保存源 YAML 的 Qwen、两个独立 DiT partition 和 Video-VAE 数值；**源库不含原 safetensors index 文件**。这些数字的证据级别是“源建模 YAML 已记录”，不能称为新仓库独立读取权重验证。Audio-VAE 仍用算子权重 proxy。单次运行只选中一个 DiT partition，不将两份权重去重。

配置字段完整快照包含已解析默认值和架构，写入 `config.snapshot.json`。更改 target 或条件但不重新测量文本行数，会得到内部可计算、却不对应真实 processor 的工作负载。

## 阶段与输出

七个阶段按源码顺序执行：Qwen vision、Qwen text、条件 VAE、text refiner、联合音视频 DiT、Video-VAE decode、Audio-VAE decode。text refiner 和联合 DiT 各执行 N 次；每步只有一个条件分支，`cfg=1`。

| 文件 | 含义 |
|---|---|
| `summary.json` | 统一导出的 plan 与 stage typed 数据；包括总时间、容量判断和执行范围 |
| `config.snapshot.json` / `provenance.json` | 有效模型/工作负载/架构快照，以及运行和来源信息 |
| `stage_metrics.csv` | 阶段总时间、加权 FLOPs/流量、调用次数、权重与激活 proxy |
| `minimax_h3_plan_metrics.csv` | 本次完整流程或所选阶段的汇总；不产生文本 token，不能用 tokens/s 解释视频生成速度 |
| 七个阶段名 `.csv` | 一个阶段可能包含多张算子图；每行有 `graph_label` 和 `graph_multiplier`，同图的原始执行成本与重复次数分开保存 |

新输出按**阶段**组织 CSV；冻结的源输出按**算子图**组织 CSV。两者通过 `graph_label` 对齐。例如 FL2VA `condition_vae_encode.csv` 包含 `condition_video_0` 和 `_1`，各执行 28 次；Ref2VA 默认参考视频图执行 224 次，视频解码图执行 196 次。重建总时延需将各图的算子行时延乘其 `graph_multiplier` 后求和，不能将 CSV 当作已经重复展开的全量 trace。

shape 在内部 workflow metadata 中同时提供 `shape_summary` 和 `resolved_shape_summary`。`operator_graph_multipliers` 标明各图次数；`packed_rows_per_second` 和 `transformer_processed_rows_per_second` 比文本 tokens/s 更适合该工作负载。

可用 `--stage transformer_denoise` 等仅分析一个阶段。所选阶段仍按“阶段 indexed 权重 + persistent tensors + peak activation proxy”判断容量，但 `memory_validity_basis=selected_stage_memory`、`request_count=null`，不推断整流程吞吐或全驻留可行性；全驻留字段为 null。完整运行默认 `assume_stage_unload=true`，使用最大阶段容量判断；`false` 使用全驻留权重、持久张量和最大激活之和。

## 算子与调度迁移

`create_workflow(hardware, config)` 使用通用 `Workflow`、`StageResult/GraphResult` 和统一导出。`bridge.py` 将模型配方变为原生 RoofLang 图，顺序控制边保留源阶段内串行累计口径；每张图只仿真一次，以 multiplier 表示去噪步数、时间窗口和 VAE tile 重复。权重在图开始时驻留，其访问流量照算；没有额外模型加载耗时。源搜索器、旧 registry 执行引擎和旧 roofline backend 没有复制。

H3 复用 MAGI 的成本算子和 analytical kernel；本包补充 Gelu、LayerNorm、Embedding。新增算子时先补参数/形状、FLOPs、activation/weight/KV 字节及计算单元，再在配方中建节点并核对数量、维度和手算结果。不要将固定时间写成算子答案。若同一算子已经在共享成本层存在，优先复用。

`hardware.py` 保留源 A5 口径：Cube FP16=486 TFLOPS×0.7、Vector FP16=54 TFLOPS×0.6；HBM 带宽使用源公式中的 `1600×0.8×1024³ bytes/s`，容量为 128 GiB；通信使用源十进制 GB/s、拓扑效率和集体操作效率及 1μs 延迟。源 HBM 与通信单位不一致，本次明确保留以复现数值。

原生设备中的 `fp16:cube` 与 `fp16:vector` 是**计算吞吐键**，不是新增张量 dtype。混合算子按两种单元的 FLOPs/吞吐耗时求和，再与访存耗时取 roofline 最大值；输出 `compute_dtype` 仍为 FP16。其他硬件需显式提供这两个吞吐键、对称的 HBM 读写带宽与容量，不能擅自把 Tensor Core 峰值用于向量运算。

## 已保留的近似与局限

- Audio ConvTranspose1D 用显式上采样后的 Conv1D 负载 proxy；Video-VAE GroupNorm 用 RMSNorm proxy；Qwen patch embed 用等价 linear 负载。源码 shape、模块顺序和配置结构与这些近似分开解释。
- FlashAttention 只计 Q/K/V/O 边界流量，未估计 tile 重读、片上容量和 kernel workspace。Qwen 视觉 attention 按文档边界分段；SP all-to-all 后联合 DiT attention 使用全局 padded 序列。
- 内存是 indexed 权重、持久张量和最大单算子 activation proxy，不是完整 tensor lifetime 或实测 allocator 峰值。允许阶段卸载不意味着卸载没有时间成本。
- FP16 是冻结工作负载精度，不重放官方 Video-VAE FP32/混合精度。缺少官方精度对应的硬件吞吐参数时不虚构换算系数。
- `execution_time_ms` 是主算子建模时延，排除媒体 resize/normalize、layout、随机数、scheduler Python、unpack、tile blending、权重搬运与 mux；不能称为实测或完整服务 E2E。当前报告的是各阶段关键设备范围，不是所有卡 FLOPs 的全局求和。
