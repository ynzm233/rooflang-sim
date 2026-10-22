# MiniMax-H3：FL2VA / Ref2VA 性能仿真

## 运行与复现

在仓库根目录安装 `pip install -e '.[test]'` 后：

```bash
rooflang-sim --model minimax_h3 --hardware A5_pr_server \
  --config examples/minimax_h3_fl2va.json --output-dir outputs/h3-fl2va
rooflang-sim --model minimax_h3 --hardware A5_pr_server \
  --config examples/minimax_h3_ref2va.json --output-dir outputs/h3-ref2va
python scripts/minimax_h3_baseline.py
python -m pytest -q tests/test_minimax_h3.py tests/test_a5_950pr.py
```

`--config` 是扁平模型参数 JSON；模型名、硬件名和输出目录通过 CLI 传入。省略 H3 硬件参数时默认 `A5_pr_server`。该模型输出性能和容量估算，不生成图片、视频、音频或模型权重。

两条默认流程均为 batch=1、SP=4、TP/CP/DP/PP=1、FP16、124 帧、768×1344、50 次去噪。采用官方 950PR 规格及下文列明的仿真假设，当前完整运行结果如下：

| 流程 | 含义 | packed rows | 当前主算子时延 ms |
|---|---|---:|---:|
| FL2VA | 首帧、尾帧或首尾两帧约束生成带音轨视频；默认首尾两帧 | 41763 | 228193.52842317586 |
| Ref2VA | 参考图像/视频和可选参考音频约束生成带音轨视频；默认参考视频带音轨 | 81532 | 685813.7461188152 |

`baselines/minimax_h3/` 保存原工程冻结的 summary、阶段 CSV、逐图算子 CSV、配置和校验信息，供历史源复现。其 `source/configs/chips/A5_pr_server.yaml` 不参与当前仿真。校验脚本通过仅用于验证的 `source_hardware()` 重放该目录，逐行比对算子身份、层号、FLOPs、访存、权重和时间分项，并核对阶段次数、总量、shape、驻留容量。当前 950PR 参数由 `tests/test_a5_950pr.py` 的手算和容量边界测试验证，默认流程结果以本页上表为准。浮点容差为相对 `1e-10`、绝对 `1e-8`；不要求不同 Python 版本产生逐字节相同的浮点字符串。

## 接入数据如何准备

输入是结构和形状数据，仿真不加载媒体或权重。`src/rooflang/programs/models/minimax_h3/architecture.json` 定义组件、层数、通道、head、FFN、patch 和权重字节数；同目录的 `workload.json` 给出完整默认请求，两份 examples 是可直接编辑的最小运行配置。

1. 准备目标 `target.frames/height/width`。宽高为 32 的正整数倍；目标帧数满足 `17*n+5`，并处于 24fps 下 5–15 秒范围。不要将 latent 尺寸写进这些像素/帧字段。
2. 准备条件。FL2VA 用 `keyframes: ["first"]`、`["last"]` 或两者；Ref2VA 用 `references`，每条含 `kind`（image/video/audio）、视觉尺寸和帧数、音频秒数。参考数量、帧数和持续时间由配置类校验；修改条件时同步下一项。
3. 从真实 Qwen processor/tokenizer 获取完整 `text_condition_rows`：包括 prompt、标签/时间戳、视觉占位、起止标记。不能只填自然语言 prompt 长度。仿真不实现 tokenizer。默认工作负载将 prompt `Generate a cinematic video.` 计为 5 tokens；FL2VA 为 `5 + 2×1016 = 2037`，Ref2VA 为 `6×1010 + 47 + 5 = 6112`。这些文本行数已写入 `workload.json`，自定义输入需要重新测量。
4. 准备运行参数 `num_inference_steps`、`sp`、`assume_stage_unload`。当前可选 SP=1/2/4/8，必须有相应设备，默认硬件含 4 卡；其余并行维度和 batch 固定为 1。冻结 profile 使用 FP16；`sglang_v0_5_21` profile 使用下文列出的组件混合精度，并要求硬件提供相应吞吐。
5. 对新结构记录配置依据和缺失部分。当前 `indexed_weight_bytes` 包含 Qwen、两个独立 DiT partition 和 Video-VAE 的权重字节数。**当前仓库不包含原始 safetensors index 文件，这些配置数值未经独立读取权重或 index 核验。** Audio-VAE 仍用算子权重 proxy。单次运行只选中一个 DiT partition，不将两份权重去重。

配置字段完整快照包含已解析默认值和架构，写入 `config.snapshot.json`。更改 target 或条件但不重新测量文本行数，会得到内部可计算、却不对应真实 processor 的工作负载。

配置可选填写 `prompt` 原文用于追溯，但必须同时显式给出 `text_condition_rows`；仿真不会自动 tokenize 或从字符串推断行数。可选 `fps` 目前只接受整数 24，同时描述参考与目标视频。

## 910B3：官方 Ref2VA 单卡与双卡

```bash
rooflang-sim --model minimax_h3 --hardware Ascend910B3_1Card \
  --config examples/minimax_h3_ref2va_910b_sp1.json \
  --output-dir outputs/minimax_h3_ref2va_910b3_sp1
rooflang-sim --model minimax_h3 --hardware Ascend910B3_2Cards \
  --config examples/minimax_h3_ref2va_910b_sp2.json \
  --output-dir outputs/minimax_h3_ref2va_910b3_sp2
```

两组使用 `execution_profile="sglang_v0_5_21"`，仅改变卡数与 SP（1／2），batch=1、TP/CP/DP/PP=1。处理后的参考视频与目标视频均为宽 1344、高 768、124 帧、24 fps。输入采用官方完整提示词、视频原音轨与独立语音；两份参考音频均按 `124/24` 秒、每声道 207 个 latent 建模。双卡场景使用 SP2，未改成部署工具包中的 TP2 方案。

### 固定请求与输入行数

[原始请求 JSON](../src/rooflang/programs/models/minimax_h3/requests/ref2va_official_d21241f.json) 保留[官方 commit `d21241f`](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/scripts/readme/reproducible-768p-ref2va-request.sh) 的完整 prompt、两个媒体 URI、target 与 seed=0。仿真消费声明的结构，不下载媒体。原始请求、脚本和 prompt 的校验值，以及逐段 token IDs，保存在[输入来源与推导](../src/rooflang/programs/models/minimax_h3/requests/ref2va_official_d21241f.provenance.json)，并由 examples 的 `request_source` 字段引用。

使用 [HF revision `42ed227` 的 tokenizer](https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/42ed227ee7df40d41602854ae760620d6eb651fe/Ref2VA/tokenizer/tokenizer.json) 和 [SGLang commit `e00930c` 的 presentation 构造](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/presentation.py) 得到完整 Qwen 输入 **6915 rows**。该路径不套用 chat template，逐段 tokenize 后拼接；提示词本身为 803 tokens、3554 UTF-8 bytes。

| 输入组成 | 推导 | rows |
|---|---|---:|
| 官方 prompt | 原文 tokenizer 结果 | 803 |
| 媒体标签 | Audio 1 为 5、Video 1 为 6、Audio 2 为 5 | 16 |
| 视频时间戳 | 6 段，每段 6 tokens | 36 |
| 视频视觉块 | 6×（1008 video pads＋2 起止标记） | 6060 |
| 完整 Qwen presentation | 上述四项之和 | **6915** |

124 帧按每 12 帧抽样得到 11 帧，末帧补齐为 12 帧，经 temporal patch=2 得到 6 块；`video_grid_thw=[6,48,84]`。块时间戳为 0.25／1.25／2.25／3.25／4.25／5.0 秒，依源码的 Python `.1f` 格式化成 `<0.2 seconds>` 至 `<5.0 seconds>`。完整序列为 `6915 + 37296 reference video + 828 reference audio + 37296 target video + 414 target audio = 82749 rows`。

上述 token IDs 经 `tokenizers 0.23.3` 独立核对，并执行锁定的 presentation 函数验证行数；验证依赖未加入运行环境。视频尺寸、音轨存在性和音频长度是本场景声明的处理后输入，尚非真实推理 trace 采集值；更换媒体或处理规则时需重新核对。

### 执行次数、精度与卸载

官方 profile 使用 Qwen 前 50 层、每请求 1 次 text refiner，以及 49 次联合 DiT forward。对应发布版的 50 个 sigma 点产生 49 次更新；CFG 蒸馏模型每步仅有一个 positive 分支。这里的仿真 `num_inference_steps=49` 表示实际更新次数，不能直接与服务请求中的 50 点参数同名比较。[锁定版去噪路径](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py#L774)

| 组件 | 计算 dtype |
|---|---|
| Qwen、text refiner、联合 DiT | BF16 |
| Video-VAE encode | FP32 |
| Video-VAE decode | FP16 |
| Audio-VAE encode／decode | FP32 |

`layerwise_offload={"prefetch_size":1,"resident_layers":0}` 显式建模 Host→HBM 权重搬运及其与计算的重叠：当前层计算时预取下一层，不保留额外常驻层。SP 分割序列，不将完整层权重按卡数平分。不同去噪步之间重新计搬运，尚未建模跨步预取；组件加载、分组与实际框架的所有内存行为也未逐一复现。本场景不启用 AdaLN 缓存、量化或跳步缓存。

逐层卸载覆盖 Qwen vision/text、text refiner、联合 DiT 和 Video-VAE decoder 的 Transformer 层；Video-VAE encoder、Audio-VAE 及非层内权重保留驻留。容量将各组件的固定权重去重相加，再加最大层缓冲区、激活代理和持久张量；不使用旧的“每阶段整模型装入／卸载”判定，`stage_unload_applied=false`。只读权重释放后无需回写主存。

### B3 规格与未校准假设

预设位于 `src/rooflang/programs/presets/ascend910b.py`。B3 型号依据华为作者的 [LocMoE+ 附录 C.2](https://arxiv.org/html/2406.00023v1#A3.SS2)：20 AI Core、1.8 GHz、FP16 总峰值约 313 TFLOPS、HBM 带宽 1600 GB/s。Cube 和 Vector 吞吐由架构分开推导，不能把总峰值同时用于两种单元。

| 参数 | 本次值 | 依据与边界 |
|---|---:|---|
| FP16／BF16 Cube、Vector | 294.912、18.432 TFLOPS | 20 cores×1.8 GHz×8192／512 FLOP/cycle；BF16 与 FP16 同吞吐是未校准假设 |
| FP32 Cube、Vector | 73.728、9.216 TFLOPS | Cube 为 FP16 的 1/4，与 [SGEMM-cube 报告的 B3 73.73 TFLOPS](https://arxiv.org/html/2507.23387v5#S6.SS1) 一致；Vector 取 FP16 的 1/2 是假设 |
| HBM 容量、带宽 | 64 GiB、1600 GB/s | 65536 MB 的容量口径参照 [B3 实机输出](https://github.com/vllm-project/vllm-ascend/issues/5056)，换算为 68,719,476,736 bytes；不扣驱动／框架保留 |
| Host→NPU DMA | 每卡每方向 32 GB/s | 理想链路假设，未实测且未核验 PCIe 配置；每卡独立 DMA，无 host root-complex 共享争用 |
| 双卡 HCCS | 每方向 28 GB/s | 将[论文中的 B4 单链路 56 GB/s](https://arxiv.org/html/2605.22428v1#S6.SS1)假设为双向合计后用于 B3，型号迁移和方向均未校准 |

效率系数为 1、固定链路延迟为 0；Host RAM 按部署工具包记录的 1.5 TiB，未扣系统保留。DMA 的 HBM 边与计算访存边独立，暂不模拟二者的共享 HBM 带宽争用。规格、推导、假设和来源均写入 `config.snapshot.json` 的 `hardware.metadata`。

### 输出适用范围

上述命令生成新的 `outputs/minimax_h3_ref2va_910b3_sp1/` 与 `outputs/minimax_h3_ref2va_910b3_sp2/` 报告。旧 910B 结果采用通用峰值、短 prompt、单份参考音频、50 步 FP16 和整阶段驻留，已不适用于当前 B3 场景。

本次重新运行结果（2026-10-20，所有时延单位为 ms）：

| 阶段／指标 | SP=1 | SP=2 |
|---|---:|---:|
| Qwen vision | 169.658 | 139.053 |
| Qwen text | 1552.646 | 1540.474 |
| Condition VAE encode | 27710.067 | 27710.067 |
| Text refiner | 76.350 | 76.350 |
| Denoise，49 次 | **2953461.839** | **1581951.855** |
| Video-VAE decode | 29824.396 | 29824.396 |
| Audio-VAE decode | 7.658 | 7.658 |
| 已建模阶段总时间 | **3012802.614** | **1641249.853** |
| Denoise 平均每步，秒 | **60.275** | **32.285** |
| 每卡容量代理峰值，GiB | **12.269** | **8.955** |
| 64 GiB 容量代理判定 | 通过 | 通过 |

每次 DiT 调用在关键 rank 搬运 64,547,507,200 bytes 层权重；49 次合计 3,162,827,852,800 bytes。按假设的 32 GB/s，串行复制工作量为 98,838.370 ms，但预取隐藏了大部分搬运；去噪阶段暴露在关键路径上的搬运等待分别约 1557.652 ms、1765.786 ms。它们已经计入阶段时延，不能再加一次。SP2 的这些字节数仍是每个关键 rank 的口径，不是两卡合计。

CSV 保留 `WeightLoad` 和零耗时 `LayerRelease` 节点；`host_to_device_bytes` 是搬运量，`copy_time_ms` 是复制工作量，`exposed_transfer_time_ms` 是未被串行计算／集合通信隐藏的时间。`weight_saved` 保留逻辑模型权重，`resident_weight_bytes` 表示该节点 weight 描述符的物理存储（`host_master` 属主存，`layer_buffer_view` 为零）；HBM 缓冲区峰值使用 `weight_peak_bytes`，完整容量代理使用 `peak_stage_memory_bytes`。

时延包含已建模主算子与逐层权重搬运。暂存权重和激活使用解析 proxy，不包含 kernel workspace、allocator 碎片和框架保留，因此容量通过不等于实机部署保证。尚缺 `conditioning_projection`、CPU 调度、实际 FlashAttention 效率与片上重读、跨步预取、媒体处理和输出封装；不得把结果称为实测完整服务耗时，也不能直接视为复现了进度条中的 126 s/it。

## 阶段与输出

七个阶段依次执行：Qwen vision、Qwen text、条件 VAE、text refiner、联合音视频 DiT、Video-VAE decode、Audio-VAE decode。冻结 profile 的 text refiner 和联合 DiT 各执行 N 次；`sglang_v0_5_21` profile 的 text refiner 每请求执行 1 次、联合 DiT 执行 N 次。每步只有一个条件分支，`cfg=1`。

| 文件 | 含义 |
|---|---|
| `summary.json` | 统一导出的 plan 与 stage typed 数据；包括总时间、容量判断和执行范围 |
| `config.snapshot.json` / `provenance.json` | 有效模型/工作负载/架构快照，以及运行标识和校验信息 |
| `stage_metrics.csv` | 阶段总时间、加权 FLOPs/流量、调用次数、权重与激活 proxy |
| `minimax_h3_plan_metrics.csv` | 本次完整流程或所选阶段的汇总；不产生文本 token，不能用 tokens/s 解释视频生成速度 |
| 七个阶段名 `.csv` | 一个阶段可能包含多张算子图；每行有 `graph_label` 和 `graph_multiplier`，同图的原始执行成本与重复次数分开保存 |

运行报告按**阶段**组织 CSV；冻结基线按**算子图**组织 CSV。两者通过 `graph_label` 对齐。例如 FL2VA `condition_vae_encode.csv` 包含 `condition_video_0` 和 `_1`，各执行 28 次；Ref2VA 默认参考视频图执行 224 次，视频解码图执行 196 次。冻结串行 profile 可将各图算子行时延乘其 `graph_multiplier` 后求和；逐层预取 profile 存在重叠，需使用各图时间线跨度乘 multiplier，或直接读取 `stage_metrics.csv`，不能累加计算与 DMA 行时延。CSV 并未按调用次数重复展开。

shape 在内部 workflow metadata 中同时提供 `shape_summary` 和 `resolved_shape_summary`。`operator_graph_multipliers` 标明各图次数；`packed_rows_per_second` 和 `transformer_processed_rows_per_second` 比文本 tokens/s 更适合该工作负载。

可用 `--stage transformer_denoise` 等仅分析一个阶段。冻结 profile 的所选阶段按“阶段 indexed 权重 + persistent tensors + peak activation proxy”判断容量；逐层卸载 profile 改用对应的暂存权重与激活 proxy。只跑一个阶段时不推断整流程吞吐或全驻留可行性，`request_count=null`。冻结 profile 完整运行默认 `assume_stage_unload=true`，使用最大阶段容量判断；`false` 使用全驻留权重、持久张量和最大激活之和。

## 算子与调度

`create_workflow(hardware, config)` 使用通用 `Workflow`、`StageResult/GraphResult` 和统一导出。`bridge.py` 将模型配方变为计算图，顺序控制边使阶段内算子按串行累计口径执行；每张图只仿真一次，以 multiplier 表示去噪步数、时间窗口和 VAE tile 重复。冻结 profile 的权重在图开始时驻留，其访问流量照算，不含额外模型加载耗时。`sglang_v0_5_21` profile 显式插入逐层权重 DMA 和预取依赖；两种场景的时延与容量口径随快照保存。

H3 复用 MAGI 的成本算子和 analytical kernel；本包补充 Gelu、LayerNorm、Embedding。新增算子时先补参数/形状、FLOPs、activation/weight/KV 字节及计算单元，再在配方中建节点并核对数量、维度和手算结果。不要将固定时间写成算子答案。若同一算子已经在共享成本层存在，优先复用。

## 硬件规格与仿真假设

`A5_pr_server` 的生效参数在 `src/rooflang/programs/models/minimax_h3/hardware.py`。选用 Ascend 950PR 的 32 Cube／64 Vector、128 GB 规格，来源为[昇腾官方处理器页](https://www.hiascend.com/hardware/processor)及其[《昇腾950 NPU架构白皮书》表3-1，PDF第13–15页](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf#page=13)，核查日期为 2026-10-15。

| 参数 | 官方峰值 | 仿真有效值 |
|---|---:|---:|
| Cube FP16 | 432 TFLOPS | 302.4 TFLOPS（×0.7） |
| Cube FP8 | 865 TFLOPS | 605.5 TFLOPS（×0.7；A5 冻结 profile 只用 FP16） |
| Vector FP16 | 54 TFLOPS | 32.4 TFLOPS（×0.6） |
| 内存容量 | 128 GB | 128×10⁹ bytes |
| 内存带宽 | 1600 GB/s | 1280 GB/s（×0.8） |

486 TFLOPS 是该 PR 规格的 Cube+Vector FP16 总算力，不能作为 Cube 单元峰值。内存容量和带宽统一按十进制计算，取消旧配置的 GiB／GiB/s 换算。

以下值缺少对应的官方依据，但会影响仿真，保留在 `SIMULATION_ASSUMPTIONS` 中：

- Cube／Vector／内存效率为 0.7／0.6／0.8。
- 单节点最多 8 卡；节点内以对称全双工逻辑边建模，带宽参数为每方向 392 GB/s，效率 0.7，延迟 1 μs。
- 集合通信按 2／4／8 卡使用 0.142857／0.428571／1 的带宽系数；其他卡数沿用最近档位。它们描述源模型的聚合通信近似，没有实测校准依据。

官方的 2016 GB/s 是芯片 UB 双向总带宽，不能直接替换这里的服务器逻辑边或集合通信带宽。以上假设并非官方规格或实测结果。它们与官方峰值、来源 URL、所选规格一起导出到 `config.snapshot.json` 的 `hardware.metadata`，可追踪有效吞吐的计算依据。

当前有效配置不包含未使用且缺少官方依据的 Vector FP8 算力、显式 HBM 零延迟和跨节点参数。历史源快照保留原样，不作为当前配置加载。当前模型不计 kernel 启动开销，也不模拟跨节点部署。

原生设备中的 `fp16:cube` 与 `fp16:vector` 是**计算吞吐键**，不是新增张量 dtype；BF16、FP32 对应同样的单元后缀。混合算子按两种单元的 FLOPs/吞吐耗时求和，再与访存耗时取 roofline 最大值；`compute_dtype` 记录该组件的计算精度。其他硬件需显式提供所用 dtype 的 Cube／Vector 吞吐、对称的 HBM 读写带宽与容量，不能擅自把矩阵单元峰值用于向量运算。

## 近似与局限

- Audio ConvTranspose1D 用显式上采样后的 Conv1D 负载 proxy；Video-VAE GroupNorm 用 RMSNorm proxy；Qwen patch embed 用等价 linear 负载。shape、模块顺序和配置结构与这些算子成本近似分别描述。
- FlashAttention 只计 Q/K/V/O 边界流量，未估计 tile 重读、片上容量和 kernel workspace。Qwen 视觉 attention 按文档边界分段；SP all-to-all 后联合 DiT attention 使用全局 padded 序列。
- 冻结 profile 的内存为 indexed 权重、持久张量和最大单算子 activation proxy；逐层卸载 profile 使用暂存权重及激活 proxy。二者均未覆盖完整 tensor lifetime、workspace、allocator 峰值和框架保留。
- FP16 是冻结工作负载精度。`sglang_v0_5_21` 按组件区分 BF16／FP32／FP16，但单组件内部的强制精度、cast 和融合行为仍需逐算子核对；B3 的 BF16 等效吞吐及 FP32 Vector 比率是明确假设。
- `execution_time_ms` 是已建模主算子时延；逐层卸载 profile 另含显式权重搬运。媒体 resize/normalize、layout、随机数、scheduler Python、unpack、tile blending 和 mux 仍未覆盖，不能称为实测或完整服务 E2E。当前报告的是各阶段关键设备范围，不是所有卡 FLOPs 的全局求和。
