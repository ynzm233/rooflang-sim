# MiniMax-H3：FL2VA / Ref2VA 性能仿真

当前模型使用独立的 Qwen 视觉／文本 FIA、联合 DiT TND attention，以及显式 Video／Audio decode kernel 配方。本文参数和结果对应当前源码；这是解析式性能仿真，不加载权重或执行真实 NPU 推理。

## 运行与复现

命令使用 Windows PowerShell，在项目根目录运行；先按 [README](../README.md#1-安装并运行) 安装依赖。多行命令的反引号后不能有空格。

```powershell
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware Ascend910B3_1Card `
  --config examples/minimax_h3_ref2va_910b_sp1.json --output-dir outputs/h3-ref2va
.\.venv\Scripts\python.exe scripts/minimax_h3_decode_validation.py `
  --measured-dir measured --output-dir outputs/minimax_h3_decode_validation
```

完整流程请显式选择 Ascend910B3 及随附配置。CLI 省略硬件时选择 `A5_pr_server`，但该预设缺少 decode 所需的 FP32 吞吐，不能完成全流程。`--stage transformer_denoise` 等参数只运行指定阶段；阶段名见下表。

## 接入数据如何准备

`architecture.json` 定义组件、层数、通道、head、FFN、patch 和权重字节；`workload.json` 提供模型默认结构输入。当前可运行 Ref2VA 配置为 [minimax_h3_ref2va_910b_sp1.json](../examples/minimax_h3_ref2va_910b_sp1.json)。

1. 设置 `target.frames/height/width`：宽高为 32 的正整数倍，目标帧数满足 `17*n+5`，在 24 fps 下持续 5–15 秒；`n` 为非负整数，字段均为像素／帧数。
2. 设置条件：FL2VA 使用 `keyframes`，Ref2VA 使用 `references` 声明图像／视频尺寸、帧数及音频秒数。仿真不会下载或处理媒体。
3. 用真实 tokenizer／processor 获取 `text_condition_rows`，包括媒体标签、时间戳、视觉占位和特殊标记；修改 prompt 时必须同步提供该计数，仿真不执行 tokenizer。
4. 明确 `execution_profile="sglang_v0_5_21"`、`num_inference_steps`、`sp` 和卸载参数。batch、TP/CP/DP/PP 均固定为 1；模型接受 SP=1/2/4/8，硬件必须提供足够设备，910B3 预设支持 1／2 卡。
5. 保存输入来源和硬件快照。`indexed_weight_bytes` 是静态配置值；仓库没有对应 safetensors index 原件，不能把它称为独立权重索引核验结果。

## 910B3：官方 Ref2VA 单卡与双卡

```powershell
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware Ascend910B3_1Card `
  --config examples/minimax_h3_ref2va_910b_sp1.json `
  --output-dir outputs/minimax_h3_ref2va_910b3_sp1_fia
```

双卡实验需复制上述 SP1 主配置，将 `sp` 改为 `2`，再将命令中的硬件改为 `Ascend910B3_2Cards`、配置路径改为自备配置路径，并使用独立输出目录。模型支持单卡 SP1 和双卡 SP2。

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
| Video-VAE decode | FP16 投影/注意力，FP32 norm/残差；Vector 计算 FP32 |
| Audio-VAE encode／decode | FP32 |

`layerwise_offload={"prefetch_size":1,"resident_layers":0}` 显式建模 Host→HBM 权重搬运及其与计算的重叠：当前层计算时预取下一层，不保留额外常驻层。SP 分割序列，不将完整层权重按卡数平分。不同去噪步之间重新计搬运，尚未建模跨步预取；组件加载、分组与实际框架的所有内存行为也未逐一复现。本场景不启用 AdaLN 缓存、量化或跳步缓存。

逐层卸载覆盖 Qwen vision/text、text refiner、联合 DiT 和 Video-VAE decoder 的 Transformer 层；Video-VAE encoder、Audio-VAE 及非层内权重保留驻留。容量将各组件的固定权重去重相加，再加最大层缓冲区、激活代理和持久张量；该配置下 `stage_unload_applied=false`。只读权重释放后无需回写主存。

### Attention 成本模型

`attention_backend="auto"` 在 Ascend910B3 上按组件选择以下成本算子；`"flash_attention"` 显式选择通用 FA。Video decode 始终使用自己的 FlashAttentionScore 配方。

| 路径 | 算子与布局 | 当前成本边界 |
|---|---|---|
| Qwen vision | `QwenVisionFIA`，非因果 TND | 一次调用保存全部视觉文档累计边界；默认 Q/KV tile=256/128，Cube 对齐 16，head dim=72 按 80 计 Cube 工作 |
| Qwen text | `QwenTextFIA`，因果 BNSD、GQA | 默认 Q/KV tile=128/512，执行模式 `full_matrix`；数学因果有效对数与全矩阵执行成本分开 |
| 联合 DiT | `NpuFusedInferAttentionScoreOperator`，TND | 序列补齐到 64 的倍数，有效文档和 padding 文档独立计算，无跨文档注意力 |
| Text refiner | `NpuFusedInferAttentionScoreOperator`，TND | 每卡完整文本，SP=1；双卡也不拆 heads/rows，不做 all-to-all |
| Video decode | `FlashAttentionScore`，BNSD | FP16 Cube／FP32 Vector，显式设备 kernel 配方 |

本例 DiT 存储行数 `82749→82752`，累计尾边界为 `[82749,82752]`，每头有效对数为 `82749²+3²`；Q/K/V/O 存储仍包含 82752 行。Text refiner 保持 6915 行、56 头与边界 `[6915]`。Qwen vision 为 6 个 4032 行文档，总计 24192 行，边界由 `qwen_attention_segments` 累加得到。

FIA 是无 `_v2` 接口的解析模型，仅支持 FP16/BF16 存储、TND/BNSD、`sparse_mode=0`、`inner_precise=0` 和无界窗口。Cube 使用输入精度，Vector 单独使用 FP32；参数校验拒绝不支持的模式。数学有效对数由 TND 累计边界差分，或由 BNSD 长度与因果标记计算。

通用 FIA 的计算工作量为：

```text
F_C = 4 × H × D × P
F_V = 15 × H × P
```

`F_C/F_V` 为 Cube／Vector FLOPs，`H` 为 Q head 数，`D` 为 head 维度，`P` 为所有文档的数学有效 QK pair 数（不含 head 维）；4 表示 QK、PV 两次矩阵乘法各计 2 FLOPs，15 表示 scale 1 加 softmax 14 的等效成本。这是解析约定，不是指令数或实测周期。

Qwen 专用模型分别记录数学有效对数、实际执行矩形和 Cube 对齐工作量：

```text
F_C = 4 × H × D_pad × P_cube
F_V = H × [15 × P_exec + (2 × D + 14) × U]
```

`D_pad` 为 D 向上对齐到 16 的维度，`P_cube` 为每个 Q 分块的行数和执行 KV 范围各向上对齐 16 后的面积之和，`P_exec` 为未对齐的实际执行矩形面积之和，`U` 为每个 Q 分块的有效行数乘额外 KV 分块数后求和，表示在线行更新次数；其余符号同上。`2D+14` 表示输出缩放／累加与在线 softmax 合并代理。三种 pair 计数均不含 head 维。

文本默认执行整个矩阵，即使数学有效对数为 `S(S+1)/2`，也不会把执行成本直接折半；`S` 是文本序列长度。可在 `qwen_fia.text.execution_mode` 选择 `causal_blocks`，仅跳过完全位于因果对角线上方的块，对角块完整执行。这是可选分析场景，未证实为实际内核分派。

`qwen_fia.vision/text` 可配置 `q_tile`、`kv_tile` 和 `cube_alignment`；tile 必须为正的 16 倍数，alignment 固定为 16。候选 tile 的来源提交和 `tiling_status=public_source_candidate_runtime_unverified` 随算子导出，尚未完成 runtime tiling 或时延校准。

单次 Q/K/V 读取归 Cube；完整 O、布尔 mask 和 FP32 LSE 边界流量归 Vector，LSE 关闭时仍计 4-byte 占位输出。Qwen 的 Cube padding 不扩大物理张量存储，边界流量不包含 tile 重读、workspace、L2 交换和 wrapper 拷贝。访存时间为：

```text
T_mem = B_C / (1600×10^9×η_C) + B_V / (1600×10^9×η_V)
```

`T_mem` 单位秒，`B_C/B_V` 为按计算单元归属的物理流量字节，1600×10⁹ 是 B3 HBM 峰值 bytes/s，`η_C=0.8`、`η_V=0.6` 是当前 Cube／Vector 访存效率。计算时间将 Cube、FP32 Vector 两项串行相加，再与访存取 Roofline 最大值。

`transformer_fia_compute_efficiency` 可覆盖联合 DiT 的 FIA Cube／Vector 计算效率，取值须在 `(0,1]`。主配置未设置此项，使用硬件预设；覆盖不作用于 Qwen、text refiner、decode、访存或 DMA。实际生效值和作用范围随输出元数据保存。

### B3 参数与未校准假设

生效参数位于 [ascend910b.py](../src/rooflang/programs/presets/ascend910b.py)。下表按当前预设计算；峰值推导、来源和假设写入 `config.snapshot.json` 的 `hardware.metadata`。

| 资源 | 峰值／物理值 | 当前效率／有效值 |
|---|---:|---:|
| FP16／BF16 Cube | 294.912 TFLOPS | 0.8／235.9296 TFLOPS |
| FP16／BF16 Vector | 18.432 TFLOPS | 0.6／11.0592 TFLOPS |
| FP32 Cube | 73.728 TFLOPS | 0.8／58.9824 TFLOPS |
| FP32 Vector | 9.216 TFLOPS | 0.6／5.5296 TFLOPS |
| HBM | 64 GiB；1600 GB/s | Cube 0.8／1280 GB/s；Vector 0.6／960 GB/s |

20 AI Core、1.8 GHz 的单元峰值推导保存在源码；BF16 与 FP16 等吞吐、FP32 Vector 比率及上述效率均为未校准假设。HBM Fabric 使用物理带宽，计算核按单元效率折减计时，不放大流量或容量。

Host RAM 为 1.5 TiB，Host↔NPU 每卡每方向 32 GB/s，双卡 HCCS 每方向 28 GB/s，链路效率为 1、固定延迟为 0。它们是显式仿真假设，不等于实测 PCIe／HCCS 性能。DMA 与计算访存使用独立 HBM 边，未模拟共享带宽或 host root-complex 竞争。

### 当前 Ref2VA 结果

2026-11-03 使用本文 SP1 命令、主配置和当前硬件预设运行；逐层卸载开启，未添加效率覆盖。数值为已建模主算子与显式权重传输的仿真时间。

| 阶段 | 当前仿真时延 ms |
|---|---:|
| qwen_vision_encode | 309.754286 |
| qwen_text_condition | 2378.230147 |
| condition_vae_encode | 34832.768220 |
| transformer_text_refiner | 100.008518 |
| transformer_denoise | 5379647.116093 |
| video_vae_decode | 30219.471941 |
| audio_vae_decode | 95.581845 |
| **完整已建模流程** | **5447582.931051** |

每卡容量代理峰值为 13187512628 bytes（12.281828 GiB），`is_valid=true`。它不包含完整 tensor lifetime、workspace、allocator 碎片和框架保留，不能作为实机部署通过结论。

`layerwise_offload=false` 可关闭逐层搬运；省略或 `null` 使用默认预取。关闭后的时间不包含首次权重加载，容量仍需按本次报告单独判断。不能把 SP 当作权重分片或合并双卡 HBM 容量。

## 阶段、输出与调度

七个阶段按上表顺序执行。当前主配置 Qwen text 使用前 50 层，text refiner 每请求一次，联合 DiT 执行 49 次，CFG=1。条件视频编码图重复 224 次；视频解码主体图重复 196 次，拼接图执行一次；音频解码图执行一次。

`create_workflow(hardware, config)` 返回通用 `Workflow`，`bridge.py` 将组件配方转换为计算图。图内计算按控制依赖串行调度，逐层 DMA 可以与计算重叠；图只仿真一次，通过 `GraphResult.multiplier` 表示重复。

`<stage>.csv` 为一次图调用的算子明细，`graph_label` 标识图，`graph_multiplier` 给出重复次数。阶段工作量按次数加权，阶段时间使用图时间线跨度加权；存在预取重叠时，不能累加计算与 DMA 行时延重建墙钟时间。权重驻留和容量峰值不乘重复次数。

`summary.json` 和 `stage_metrics.csv` 保存计划／阶段指标；`config.snapshot.json` 与 `provenance.json` 保存生效配置、硬件和包哈希。单阶段运行仅代表所选范围，`request_count=null`，不能推断整流程吞吐。字段见 [输出参考](output_reference.md)。

## 硬件规格与仿真假设

`A5_pr_server` 的生效参数在 `src/rooflang/programs/models/minimax_h3/hardware.py`。选用 Ascend 950PR 的 32 Cube／64 Vector、128 GB 规格，来源为[昇腾官方处理器页](https://www.hiascend.com/hardware/processor)及其[《昇腾950 NPU架构白皮书》表3-1，PDF第13–15页](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf#page=13)，核查日期为 2026-10-15。

| 参数 | 官方峰值 | 仿真有效值 |
|---|---:|---:|
| Cube FP16 | 432 TFLOPS | 302.4 TFLOPS（×0.7） |
| Cube FP8 | 865 TFLOPS | 605.5 TFLOPS（×0.7） |
| Vector FP16 | 54 TFLOPS | 32.4 TFLOPS（×0.6） |
| 内存容量 | 128 GB | 128×10⁹ bytes |
| 内存带宽 | 1600 GB/s | 1280 GB/s（×0.8） |

486 TFLOPS 是该 PR 规格的 Cube+Vector FP16 总算力，不能作为 Cube 单元峰值。内存容量和带宽统一按十进制计算。

以下值缺少对应的官方依据，但会影响仿真，保留在 `SIMULATION_ASSUMPTIONS` 中：

- Cube／Vector／内存效率为 0.7／0.6／0.8。
- 单节点最多 8 卡；节点内以对称全双工逻辑边建模，带宽参数为每方向 392 GB/s，效率 0.7，延迟 1 μs。
- 集合通信按 2／4／8 卡使用 0.142857／0.428571／1 的带宽系数；其他卡数沿用最近档位。它们描述源模型的聚合通信近似，没有实测校准依据。

官方的 2016 GB/s 是芯片 UB 双向总带宽，不能直接替换这里的服务器逻辑边或集合通信带宽。以上假设并非官方规格或实测结果。它们与官方峰值、来源 URL、所选规格一起导出到 `config.snapshot.json` 的 `hardware.metadata`，可追踪有效吞吐的计算依据。

A5 未提供 BF16 和 FP32 Cube／Vector 吞吐，不能运行本文的混合精度 Ref2VA 配置或完整 decode。当前模型不计 kernel 启动固定开销，也不模拟跨节点部署。

原生设备中的 `fp16:cube` 与 `fp16:vector` 是**计算吞吐键**，不是新增张量 dtype；BF16、FP32 对应同样的单元后缀。混合算子按两种单元的 FLOPs/吞吐耗时求和，再与访存耗时取 roofline 最大值；`compute_dtype` 记录该组件的计算精度。其他硬件需显式提供所用 dtype 的 Cube／Vector 吞吐、对称的 HBM 读写带宽与容量，不能擅自把矩阵单元峰值用于向量运算。

## 近似与局限

- Qwen tile、FIA 和音频转置卷积是解析候选实现；Cube／Vector 的真实流水重叠、缓存和 kernel 效率尚未校准。通用 FA 与 FIA 的边界流量不代表完整实测 HBM 流量。
- Video／Audio decode 的类型、形状和调用次数与提供的设备明细核对，成本仍由公式计算。当前误差及音频分块模型见 [decode 建模说明](minimax_h3_decode.md)。
- Video-VAE encoder 的 norm、Qwen patch embed 与容量仍有近似；部分组件内部的 cast／融合行为未逐核对齐。
- `execution_time_ms` 不包含完整媒体处理、conditioning projection、CPU 调度、kernel launch 固定开销、scheduler、封装与 mux。统计范围是各阶段关键 rank，不能乘卡数解释成全局工作量。
- 结构、算术与本地测试通过不等于 NPU 时延验收；缺失的实测项保持未知，不设置未经授权的误差通过阈值。
