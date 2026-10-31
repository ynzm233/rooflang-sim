# MiniMax-H3：FL2VA / Ref2VA 性能仿真

命令使用 Windows PowerShell；请先按 [README](../README.md#1-安装并运行) 创建虚拟环境并安装依赖，再在项目根目录执行。多行命令末尾的反引号（`` ` ``）后不能有空格。

## 910B3 通用 FA 重跑（2026-10-28）

配置显式设置 `"attention_backend": "flash_attention"`，所有 attention 使用仓库原有
`FlashAttentionOperator` 成本模型。省略该字段或设为 `"auto"` 保留原有硬件选择行为
（910B3 使用 FIA）。通用 FA 沿用原有 SP 长度对齐，不应用 FIA 分支的 64 行 padding；
910B3 的 text refiner 仍在每卡处理完整提示词、SP=1。

本次输入沿用仓库锁定的官方 Ref2VA 请求，参考视频、原音轨及独立语音保持不变；
输出宽 1344、高 768、124 帧、24 fps。使用 `sglang_v0_5_21`、BF16、49 次去噪，
逐层卸载 `prefetch_size=1, resident_layers=0`。

```powershell
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware Ascend910B3_1Card `
  --config examples/minimax_h3_ref2va_910b_sp1_fa.json `
  --output-dir outputs/minimax_h3_ref2va_910b3_sp1_fa
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware Ascend910B3_2Cards `
  --config examples/minimax_h3_ref2va_910b_sp2_fa.json `
  --output-dir outputs/minimax_h3_ref2va_910b3_sp2_fa
```

| 配置 | 建模总耗时 | 去噪耗时 | 每卡峰值显存代理 | 容量判定 |
|---|---:|---:|---:|---|
| 单卡 SP1 | 3012.803 s（50.213 min） | 2953.462 s | 12.269 GiB | 通过 |
| 双卡 SP2 | 1641.250 s（27.354 min） | 1581.952 s | 8.955 GiB | 通过 |

双卡相对单卡加速 1.836×。两组算子 CSV 各包含 165 条通用 FA 记录、0 条 FIA 记录；
图重复次数另存于 `graph_multiplier`。输出目录包含配置快照、逐算子 CSV、阶段指标、
`summary.json`、`validation.json` 和运行／测试日志；相关测试 116 项通过。
这些数值是未校准的主算子及逐层权重传输估算，不是实机端到端时延；显存判定沿用仓库的
权重、持久张量和峰值算子 activation 代理口径。

## 运行与复现

在仓库根目录安装 `pip install -e '.[test]'` 后：

```powershell
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware A5_pr_server `
  --config examples/minimax_h3_fl2va.json --output-dir outputs/h3-fl2va
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware A5_pr_server `
  --config examples/minimax_h3_ref2va.json --output-dir outputs/h3-ref2va
.\.venv\Scripts\python.exe scripts/minimax_h3_baseline.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_minimax_h3.py tests/test_a5_950pr.py
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

```powershell
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware Ascend910B3_1Card `
  --config examples/minimax_h3_ref2va_910b_sp1.json `
  --output-dir outputs/minimax_h3_ref2va_910b3_sp1_fia
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware Ascend910B3_2Cards `
  --config examples/minimax_h3_ref2va_910b_sp2.json `
  --output-dir outputs/minimax_h3_ref2va_910b3_sp2_fia
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

### 昇腾融合 Attention 成本算子

选择 `Ascend910B3`（含兼容别名 `Ascend910B`）时，MiniMax-H3 图中的全部 `FlashAttentionOperator` 自动替换为 `NpuFusedInferAttentionScoreOperator`。参数类与算子从 `rooflang.programs.models.magi2_preview.operators` 导出；实现位于 `_cost/npu_fused_infer_attention_score.py`。其它硬件与模型继续使用原算子，历史 A5 基线不变。

这是 `torch.ops.npu.npu_fused_infer_attention_score` **无 `_v2` 接口的解析模型**，不会执行 NPU kernel，也不表示 FA3。参数保留原生接口命名：`num_heads`、`num_key_value_heads`（0 表示 MHA）、`input_layout`、`actual_seq_lengths`、`actual_seq_lengths_kv`、`scale`、`sparse_mode`、`pre_tokens`、`next_tokens`、`inner_precise`、`softmax_lse_flag`。额外的 `batch_size/q_len/kv_len/head_dim` 描述张量存储，`causal` 描述显式布尔三角掩码。当前仅建模 FP16/BF16、TND/BNSD、`sparse_mode=0`、`inner_precise=0`、无界窗口（2147483647）；未支持模式会报错，量化、分页 KV 和任意稀疏掩码不在此参数子集中。[官方 API 语义](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md)

| 路径 | 布局与长度 | 成本边界 |
|---|---|---|
| 联合 DiT（FL2VA／Ref2VA） | TND；`q_len/kv_len` 为总存储行数；actual lengths 为累计尾边界 | 有效文档与 padding 文档分别计算，无跨文档注意力 |
| Text refiner | TND；完整文本，空尾段被过滤 | 每 rank 完整计算，SP2 也不拆 heads/rows、不做 all-to-all |
| Qwen vision、Video-VAE decoder | BNSD；长度为每 batch 存储行数 | 保留视觉文档分段及原有 batch、heads；这是统一 FIA 的建模选择 |
| Qwen text | BNSD、GQA、显式 `(1,S,S)` 布尔因果掩码 | 数学有效对数为 `S(S+1)/2`；计掩码读取，K/V 为本次调用输入，不推断持久缓存 |

锁定 [SGLang `e00930c` 的 Ascend 包装](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/multimodal_gen/runtime/layers/attention/backends/ascend_fa.py#L92)确认 DiT 的 TND 接口、`scale=D**-0.5` 和 `softmax_lse_flag=False`。[packed_sequence.py](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py#L385) 将联合序列补齐到 64 倍数，构造 `[0,used,padded]`。本例 `82749→82752`，实际传入 `[82749,82752]`，每头有效对数为 `82749²+3²`，不是 `82752²`。Q/K/V/O 仍包含全部 82752 行。Text refiner 保持 6915 行、56 头和 `[6915]`；[固定模型源码](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1347)未对它启用 Ulysses。

所有其它 FA 也按请求统一使用 FIA 成本模型，但 Qwen/VAE 的真实部署分派未逐一核验；CSV 的 `attention_alignment=analytical_fia_substitution` 明确标记这一点。D=72/64 的路径使用 BNSD，不能直接套用 DiT 的 TND 配置。CSV 的 `shape` 保存完整参数，另有 `source_operator`、`atten_mask_shape`、`attention_pairs` 和 `assumption`；配置快照与计划指标的 `attention_model` 保存选择与未校准状态。

2026-10-30 按 [CANN 官方 FIA 文档](https://gitee.com/ascend/cann-ops-adv/blob/master/docs/FusedInferAttentionScore.md)重构：`inner_precise=0` 为高精度路径，Cube 保持输入 FP16/BF16，Vector 单独使用 FP32，不再共享 dtype。官方 [类型定义](https://gitee.com/ascend/cann-ops-adv/blob/master/src/transformer/prompt_flash_attention/prompt_flash_attention_base.h)中 BF16 和 FP16 HighPrecision 的 `mmOutputType/softmaxType` 均为 `float`；这不代表 FP16 HighPerformance 路径也采用同样精度。

计算按 `4×H×D×有效对数` Cube FLOPs 和 `15×H×有效对数` FP32 Vector FLOPs 估算。15 包含 scale 乘法 1 和既有 Softmax 14（exp=10、sub/div 各 1、max/sum reduction 各 1）；官方 [x910 内核](https://gitee.com/ascend/cann-ops-adv/blob/master/src/transformer/prompt_flash_attention/prompt_flash_attention_s1s2_bns1_x910.h)在 Softmax 前显式执行 `Muls(scaleValue)`。这些 FLOPs 是解析成本约定，不是 CANN 指令数或实测周期。Cube/Vector 计算时间仍串行相加后与访存时间取 max，未模拟真实流水重叠、在线 Softmax 更新和转换开销。

访存保留原输入 dtype：单次 Q/K/V 读取归 Cube，完整 O、布尔 mask、FP32 LSE 写出/读取归 Vector；LSE 关闭时仍计 4-byte 占位输出。以此明确的边界流量假设计算 `Tmem=bytes_cube/(1600GB/s×eff_cube)+bytes_vector/(1600GB/s×eff_vector)`，当前两项效率均为 0.6。不放大真实字节数，也不把全部中间矩阵直接算作 HBM。官方部分 x910 模板会通过 GM workspace 交换分块 score/probability，其实际 HBM 流量还依赖 L2；本模型未覆盖这部分、tile 重读、布局转换和掩码构造，因此不能把边界流量理解为完整实测 HBM 流量。

CSV 额外导出 `cube_compute_dtype/vector_compute_dtype`、`flops_by_dtype`、`memory_bytes_by_unit`、`memory_efficiency_by_unit` 和 `local_memory_efficiency`。后者为按字节加权的有效效率，只用于计时，流量与容量代理不随效率变化。B3 的其它纯 Cube/Vector 算子也应用各自效率，当前均为 0.6；显式选择通用 FA 等未提供访存分项的 mixed 算子，融合边界按 Cube 归属并写入 `memory_efficiency_basis`。

### 关闭逐层卸载的 SP2 场景

如需关闭逐层卸载，官方 profile 支持显式 `"layerwise_offload": false`；省略或 `null` 仍使用默认逐层预取。可复现的 SP2 配置为 `examples/minimax_h3_ref2va_910b_sp2_no_offload.json`，只改变卸载选项，保留 BF16、49 步及 FIA：

```powershell
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware Ascend910B3_2Cards `
  --config examples/minimax_h3_ref2va_910b_sp2_no_offload.json `
  --output-dir outputs/minimax_h3_ref2va_910b3_sp2_fia_no_offload
```

2026-10-27 重跑：无 `WeightLoad`/`LayerRelease`，165 个 FA 均为 FIA；逐层 Host→HBM 搬运为 0。预加载权重后的已建模总时间为 1617.111 秒，DiT 49 步为 1580.168 秒（32.248 秒/步）。全部组件常驻时每卡权重约 115.351 GiB、容量代理峰值 118.778 GiB，超过 64 GiB，因此报告 `is_valid=false`。SP 不分摊本模型的权重；这里未启用阶段卸载，也未把双卡 HBM 合并。这些时间仅供理论比较，不含首次加载，不能视为可部署方案的实测耗时。

### B3 规格与未校准假设

预设位于 `src/rooflang/programs/presets/ascend910b.py`。B3 型号依据华为作者的 [LocMoE+ 附录 C.2](https://arxiv.org/html/2406.00023v1#A3.SS2)：20 AI Core、1.8 GHz、FP16 总峰值约 313 TFLOPS、HBM 带宽 1600 GB/s。Cube 和 Vector 吞吐由架构分开推导，不能把总峰值同时用于两种单元。

| 参数 | 物理／推导峰值 | 依据与边界 |
|---|---:|---|
| FP16／BF16 Cube、Vector | 294.912、18.432 TFLOPS | 20 cores×1.8 GHz×8192／512 FLOP/cycle；BF16 与 FP16 同吞吐是未校准假设 |
| FP32 Cube、Vector | 73.728、9.216 TFLOPS | Cube 为 FP16 的 1/4，与 [SGEMM-cube 报告的 B3 73.73 TFLOPS](https://arxiv.org/html/2507.23387v5#S6.SS1) 一致；Vector 取 FP16 的 1/2 是假设 |
| HBM 容量、带宽 | 64 GiB、1600 GB/s | 65536 MB 的容量口径参照 [B3 实机输出](https://github.com/vllm-project/vllm-ascend/issues/5056)，换算为 68,719,476,736 bytes；不扣驱动／框架保留 |
| Host→NPU DMA | 每卡每方向 32 GB/s | 理想链路假设，未实测且未核验 PCIe 配置；每卡独立 DMA，无 host root-complex 共享争用 |
| 双卡 HCCS | 每方向 28 GB/s | 将[论文中的 B4 单链路 56 GB/s](https://arxiv.org/html/2605.22428v1#S6.SS1)假设为双向合计后用于 B3，型号迁移和方向均未校准 |

2026-10-31 起 B3 默认 Cube/Vector 计算效率均为 **0.7**，MiniMax-H3 的 Cube/Vector 访存效率均为 **0.6**（Vector 从 0.3 调整为 0.6）。对应 FP16/BF16 Cube 有效算力 206.4384 TFLOPS、FP32 Vector 6.4512 TFLOPS，两类算子的有效访存带宽均为 960 GB/s。这些是用户指定的假设，不是官方性能保证或实测校准。HBM Fabric 保留物理 1600 GB/s（全局 `memory_efficiency=1`），每个计算节点单独折减本地访存时间；DMA 和 Host 链路不套用 Cube/Vector 效率。

固定链路延迟为 0；Host RAM 按部署工具包记录的 1.5 TiB，未扣系统保留。DMA 的 HBM 边与计算访存边独立，暂不模拟二者的共享 HBM 带宽争用。规格、推导、效率、假设和来源均写入 `config.snapshot.json` 的 `hardware.metadata`。

### Vector 访存效率 0.6 的单卡 Ref2VA 结果（2026-10-31）

相对下方 2026-10-30 结果，仅将 Vector 访存效率从 0.3 提高至 0.6。沿用单卡 SP1、FIA FP32 Vector、49 次去噪、逐层卸载 `prefetch_size=1/resident_layers=0`，输入配置和其它硬件参数均保持一致。

```powershell
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware Ascend910B3_1Card `
  --config examples/minimax_h3_ref2va_910b_sp1.json `
  --output-dir outputs/minimax_h3_ref2va_910b3_sp1_fia_fp32_vector_mem06_20261031
```

| 阶段／指标 | Vector 访存效率 0.3，秒 | Vector 访存效率 0.6，秒 |
|---|---:|---:|
| Qwen vision | 0.334 | 0.299 |
| Qwen text | 2.344 | 2.188 |
| Condition VAE encode | 40.512 | 39.731 |
| Text refiner | 0.111 | 0.106 |
| Denoise，49 次 | 5528.940 | **5401.780** |
| Video-VAE decode | 30.012 | 29.955 |
| Audio-VAE decode | 0.011 | 0.011 |
| 已建模阶段总时间 | 5602.264 | **5474.070** |
| Denoise 平均每步 | 112.836 | **110.240** |

总时间减少 **128.193 秒（2.288%）**。FLOPs、物理访存字节数和容量代理均未变化；峰值仍为 **12.270 GiB**，64 GiB 容量代理检查通过。去噪计算工作量时间仍为 5276.501 秒，访存工作量时间从 315.148 秒下降至 184.506 秒；算子按 roofline 取计算与访存的瓶颈，因此不能把访存工作量的减少直接视为同等阶段加速。上述工作量与阶段时延不可重复相加。

输出目录保留 `comparison.json`、完整阶段/算子 CSV、`config.snapshot.json`、运行日志和本次源码差异。比较文件核验输入配置、注意力模型、其它硬件参数、FLOPs、流量及容量代理均保持一致。时延仍为解析估算，未完整覆盖内部 workspace/L2 与服务开销。

### 历史：FP32 Vector 与指定效率的单卡 Ref2VA 结果（2026-10-30）

沿用官方请求 SP1、BF16 输入、124 帧 1344×768、49 次去噪及逐层卸载 `prefetch_size=1/resident_layers=0`。此历史结果采用 Vector 访存效率 0.3；以下命令应在提交 `637d200` 上执行，当前默认效率已更新为 0.6：

```powershell
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware Ascend910B3_1Card `
  --config examples/minimax_h3_ref2va_910b_sp1.json `
  --output-dir outputs/minimax_h3_ref2va_910b3_sp1_fia_fp32_20261030
```

| 阶段／指标 | 改前（秒） | 本次（秒） |
|---|---:|---:|
| Qwen vision | 0.170 | 0.334 |
| Qwen text | 1.553 | 2.344 |
| Condition VAE encode | 27.710 | 40.512 |
| Text refiner | 0.076 | 0.111 |
| Denoise，49 次 | 2953.484 | **5528.940** |
| Video-VAE decode | 29.824 | 30.012 |
| Audio-VAE decode | 0.008 | 0.011 |
| 已建模阶段总时间 | 3012.825 | **5602.264** |
| Denoise 平均每步 | 60.275 | **112.836** |

总时间增加 **85.947%（1.859×）**，同时包含精度、scale FLOPs、计算效率和访存效率四项变化，不能全部归因于 FP32。旧源 `f3b54c866b13787f9e00ea8a325ec094e7693195` 已通过独立 `git archive` 目录复跑，基线位于 `outputs/minimax_h3_ref2va_910b3_sp1_fia_before_20261030`；本次目录保留 `comparison.json`、`source-changes.patch`、完整 CSV、配置快照和源码校验信息。

全部 165 行 FIA 均核验为 FP16/BF16 Cube + FP32 Vector；所有阶段的真实流量、Host 搬运字节数和容量代理均与改前一致。容量代理峰值 **12.270 GiB**，64 GiB 检查通过。去噪 Host→HBM 仍为 3,162,827,852,800 bytes，串行复制工作量 98.838 秒，其中暴露等待约 0.839 秒；等待已包含在时延内。更长的计算隐藏了更多复制时间。

验证：全量测试 **775 passed、1 skipped**，包含独立算术检查和历史 A5 基线；`outputs/fia-full-tests.log` 保留输出。结果是未完整覆盖 workspace/L2、真实流水和服务开销的解析估算，容量通过仍是代理判定。

### 相同条件下关闭逐层卸载（单卡）

`examples/minimax_h3_ref2va_910b_sp1_no_offload.json` 从单卡逐层卸载运行的输入快照复制，仅将 `layerwise_offload` 改为 `false`。FIA FP32 Vector、49 步及其它请求条件保持一致，也未启用阶段卸载；效率采用当前 B3 预设（计算 0.7、Cube/Vector 访存均为 0.6）。

```powershell
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware Ascend910B3_1Card `
  --config examples/minimax_h3_ref2va_910b_sp1_no_offload.json `
  --output-dir outputs/minimax_h3_ref2va_910b3_sp1_fia_fp32_no_offload
```

无卸载结果需要重新运行生成；检查 `summary.json` 的容量判定后再使用时延。此模式假设权重已预加载，不含首次加载时间。

### 输出适用范围

上述命令生成 `outputs/minimax_h3_ref2va_910b3_sp1_fia/` 与 `outputs/minimax_h3_ref2va_910b3_sp2_fia/` 报告。未带 `_fia` 的 B3 目录是改用 FIA 之前的结果；更早的 910B 结果还采用通用峰值、短 prompt、单份参考音频、50 步 FP16 和整阶段驻留。

历史 FIA 成本模型结果（2026-10-27，Cube/Vector 共享输入 dtype、Softmax14、效率全 1；所有时延单位为 ms，不代表当前默认参数）：

| 阶段／指标 | SP=1 | SP=2 |
|---|---:|---:|
| Qwen vision | 169.658 | 139.053 |
| Qwen text | 1552.646 | 1540.474 |
| Condition VAE encode | 27710.067 | 27710.067 |
| Text refiner | 76.350 | 76.350 |
| Denoise，49 次 | **2953483.816** | **1581933.357** |
| Video-VAE decode | 29824.396 | 29824.396 |
| Audio-VAE decode | 7.658 | 7.658 |
| 已建模阶段总时间 | **3012824.591** | **1641231.354** |
| Denoise 平均每步，秒 | **60.275** | **32.284** |
| 每卡容量代理峰值，GiB | **12.270** | **8.955** |
| 64 GiB 容量代理判定 | 通过 | 通过 |

每次 DiT 调用在关键 rank 搬运 64,547,507,200 bytes 层权重；49 次合计 3,162,827,852,800 bytes。按假设的 32 GB/s，串行复制工作量为 98,838.370 ms，但预取隐藏了大部分搬运；去噪阶段暴露在关键路径上的搬运等待分别约 1557.640 ms、1765.782 ms。它们已经计入阶段时延，不能再加一次。SP2 的这些字节数仍是每个关键 rank 的口径，不是两卡合计。

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
- 通用 FlashAttention 只计 Q/K/V/O 边界流量；910B3 的 FIA 模型另计显式 mask/LSE，并把联合 DiT 的有效序列和尾部 padding 分开计算注意力。两者均未估计 tile 重读、片上容量和 kernel workspace；Qwen 视觉 attention 保留文档边界。
- 冻结 profile 的内存为 indexed 权重、持久张量和最大单算子 activation proxy；逐层卸载 profile 使用暂存权重及激活 proxy。二者均未覆盖完整 tensor lifetime、workspace、allocator 峰值和框架保留。
- FP16 是冻结工作负载精度。`sglang_v0_5_21` 按组件区分 BF16／FP32／FP16，但单组件内部的强制精度、cast 和融合行为仍需逐算子核对；B3 的 BF16 等效吞吐及 FP32 Vector 比率是明确假设。
- `execution_time_ms` 是已建模主算子时延；逐层卸载 profile 另含显式权重搬运。媒体 resize/normalize、layout、随机数、scheduler Python、unpack、tile blending 和 mux 仍未覆盖，不能称为实测或完整服务 E2E。当前报告的是各阶段关键设备范围，不是所有卡 FLOPs 的全局求和。
