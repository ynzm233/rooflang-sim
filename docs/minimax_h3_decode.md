# MiniMax-H3 decode：实测结构重构

2026-11-02。Video 和 Audio decode 已替换为显式设备 kernel 配方，旧 decode block、通用 Transformer 拼装和 ConvTranspose 的普通卷积代理已删除，没有旧模型切换入口。其它阶段保留原实现。

模型从实测恢复算子类型、张量形状、精度和重复次数；运行时仍由硬件参数与解析工作量计算时间，不读取实测时延。**结构对齐不等于时延已校准：音频转置卷积仍有明显低估。**

## 数据与验证结果

来源为仓库 `measured/video_vae_decode.csv`、`measured/audio_vae_decode.csv` 和 `measured/stage_summary.csv`。前两份是按设备 kernel 签名聚合的明细，非模块级父子调用列表；各行 calls 和 total_ms 的总和分别与阶段汇总一致。验证脚本保存文件 SHA-256、完整请求配置与硬件快照。

本次比较使用 `examples/minimax_h3_ref2va_910b_sp1.json`，只将 `layerwise_offload` 设为 `false`，消除模拟 H2D 与 kernel 总时长之间的口径差异。124 帧、768×1344、音频 latent 长度 207，与明细中的尺寸一致。CSV 未提供设备型号；采用当前 Ascend910B3 单卡预设是比较假设，不能由 CSV 独立证实。

| 阶段 | 实测 kernel 次数 | 模拟 kernel 次数 | 实测 kernel ms | 模拟 kernel ms | 有符号误差 |
|---|---:|---:|---:|---:|---:|
| Video decode | 459714 | 459714 | 20360.597090 | 18313.903744 | -10.05% |
| Audio decode | 4590 | 4590 | 947.835833 | 94.115983 | -90.07% |

误差定义为 `100 × (S − M) / M`。`S` 是模拟设备 kernel 总时长，单位 ms；`M` 是实测设备 kernel 总时长，单位 ms；`100` 将比例转换为百分比，负值表示模拟低估。只展示误差，不设通过阈值。

Video 的全部 101 种规范化浮点张量签名及重复次数匹配。签名包含 op_type、输入/输出形状及 dtype，合并权重输入，忽略未提供实际张量的标量/元数据；原 CSV 有 108 行，部分行仅 kernel_name 不同。Audio 的 4336 个有形状记录的调用逐项匹配输入、输出、精度和次数，包括 INT32/INT64 元数据；另 254 个 MemSet 在原始 CSV 中没有形状，流量使用明确标注的目标滤波器缓冲区估算。

实测 video `device_span=51584.614440 ms`，audio `device_span=974.030121 ms`。跨度包含 kernel 间空隙，与 kernel 总时长不同；host_total 也单列，不能叠加到 kernel_total。模型未重建主机调度或这些空隙。

## Video 配方

`video_decode.py` 分为每 tile/window 的主体图和每请求一次的拼接图。

| 图 | 单图 kernel 数 | 默认重复次数 | 工作 |
|---|---:|---:|---|
| `video_vae_decode` | 2332 | 196 | 输入卷积/投影、36 层 Transformer、输出投影与 unpatchify |
| `video_vae_decode_stitch` | 2642 | 1 | 空间重叠混合、裁切、时间拼接、ViewCopy |

每个主体图为 `36 × 64 + 28 = 2332` 次 kernel 调用。`36` 是 Transformer 层数，`64` 是每层 kernel 数，`28` 是该图层外输入/输出处理调用数；所有数字单位均为调用次数或层数，不能解释为时延。

每层包含 4 次 MatMul、1 次 FlashAttentionScore、1 次 SwiGlu，以及分解的 RMSNorm/QKNorm、48 维部分 RoPE、残差缩放、Cast、Slice、Transpose、AsStrided 和 Concat。注意力矩阵乘法及投影使用 FP16，归一化与残差存在 FP32 张量；Vector 工作量使用 FP32 吞吐。设备调用确认为 FlashAttentionScore，不再受 DiT 的 `attention_backend` FIA 替换影响。输出投影在 36 层之后执行。

主体图序列长为 `S = 7 × (h / 16) × (w / 16) + 5`。`S` 是 token 数；`h`、`w` 是截断到不超过 256 的 tile 高、宽，单位像素；`7` 是每次解码的 latent 帧数；`16` 是每个空间维度的像素压缩倍数；`5` 是额外 token 数。默认 `h=w=256`，所以 `S=1797`。

默认空间 tile 数为 28，时间窗口数为 7，主体图重复 `28 × 7 = 196` 次。每层注意力调用数为 `36 × 196 = 7056`；权重驻留量按唯一配方计一次，不乘 196。逐层卸载启用时，36 层的 DMA 按每次图调用重复，拼接图不加载层权重。

注意力工作量为 `F_C = 4 × H × S² × D`、`F_V = 15 × H × S²`。`F_C`、`F_V` 分别是 Cube 和 Vector 的 FLOP 数；`H=32` 是 head 数；`S` 是序列 token 数；`D=64` 是每个 head 的维度；`4` 来自 QK 与 PV 两次矩阵乘法各按乘加 2 FLOPs；`15=1+14` 是每个有效注意力 pair 的缩放和 softmax 成本约定。此处 batch 为 1、非因果注意力，无三角折半。

空间拼接按照 16 像素对齐分配重叠量，在默认输入下恢复垂直 80/80/96、水平 80/80/80/80/64/64 像素的重叠直方图。其它分辨率的均衡分配是外推假设；聚合 CSV 不提供相邻 tile 的真实执行次序。Slice/AsStrided 只读所选区域，ViewCopy 只写目标区域，不把整个 backing allocation 重复当作流量。

## Audio 配方

`audio_decode.py` 保留立体声 batch=2，所有实测浮点张量为 FP32。结构是输入投影、ConvPre、7 次可学习转置卷积上采样、126 次残差卷积、ConvPost 与输出裁剪；没有证据的注意力或 pre-transformer 不加入图中。

上采样率依次为 `[5,5,2,2,2,2,2]`，乘积为 800；输出采样点数为 `L_out = 800 × L_0`。`L_out` 是每声道输出采样点数；`L_0` 是每声道输入 latent 数，默认 207；`800` 是七级上采样率的乘积，默认输出 165600 点。

每次残差卷积前，以及 ConvPost 前，都有一次抗混叠 Snake，共 127 组。每组包括 padding、12-tap 分组转置卷积上采样、Snake 的 Exp/Reciprocal/Sin/Pows/Mul/Add、切片与低通下采样。加上 7 个可学习上采样器，总计 134 次 Conv2DTranspose。还显式建模权重归一化、NC1HWC0/滤波器布局转换与清零。

普通卷积工作量为 `F = 2 × B × L_out × C_out × (C_in / G) × K`。`F` 是 FLOP 数；`B=2` 是声道并入的 batch；`L_out` 是该层输出长度；`C_in`、`C_out` 是输入、输出通道数；`G` 是分组数；`K` 是有效卷积核 tap 数；`2` 表示一次乘加计 2 FLOPs。dilation 改变采样位置与 padding，不扩大有效 tap 数。bias 另计每个输出元素一次加法。

转置卷积工作量为 `F = 2 × B × L_in × C_in × (C_out / G) × K`。符号含义同上，`L_in` 是实际输入长度；该估算包含边缘被裁去的 tap，不将上采样后的输出长度错误代入输入工作量，也不因零填充虚构乘加。

物理流量按实测 blocked tensor 的 shape 和 dtype 计算；padding 后的物理容量与逻辑 FLOPs 分开。已归一化/转换的临时权重保留读取流量，不重复计入永久驻留权重。

实测 134 次 Conv2DTranspose 总计 867.372689 ms，模拟约 4.184930 ms，贡献了主要误差。数学 FLOPs、张量字节数与统一硬件效率无法解释该实现的低利用率。当前没有把这段慢时间改写成虚构 FLOPs，也没有校准特定 shape 的有效吞吐。**音频时延目前仍不准确，不能据此声称 decode 已达到实测精度。**

## 计算、流量与范围

每个张量的物理字节数为 `Q = b × ∏ d_i`。`Q` 是字节数；`b` 是存储 dtype 的每元素字节数，FP16=2、FP32=4、INT32=4、INT64=8；`d_i` 是第 `i` 个维度的元素数；`∏` 对全部维度取乘积，标量为一个元素。读取与写入分别累加，视图使用实际触及区域。

每个 kernel 的解析时间采用现有 Roofline/离散事件模拟器：混合计算按各单元工作量和对应 dtype 吞吐累计，再与本地访存时间比较。Cube/Vector 效率沿用硬件预设，不从实测时延反推；默认 B3 两种访存效率均为 0.6。新增模型没有全局调低其它阶段的吞吐。

本次捕获确认的是 kernel 边界与尺寸分布。kernel launch 固定开销、workspace/L2 流量、真实分组卷积算法效率、主机同步和准确的激活存活期仍未建模；显存结果仍是代理量。其它输入尺寸按配方外推，并非额外实测点验证。

## 运行与文件

Windows PowerShell，在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts/minimax_h3_decode_validation.py `
  --measured-dir measured --output-dir outputs/minimax_h3_decode_validation_20261102
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware Ascend910B3_1Card `
  --config examples/minimax_h3_ref2va_910b_sp1.json --stage video_vae_decode `
  --output-dir outputs/minimax_h3_video_decode
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware Ascend910B3_1Card `
  --config examples/minimax_h3_ref2va_910b_sp1.json --stage audio_vae_decode `
  --output-dir outputs/minimax_h3_audio_decode
```

macOS/Linux 的验证命令为 `.venv/bin/python scripts/minimax_h3_decode_validation.py --output-dir outputs/minimax_h3_decode_validation_20261102`。

验证目录中的 `decode_validation.md`/`.json` 保存时延、调用数、逐类型差额、来源 SHA-256；`simulation/<stage>/` 保存配置快照、阶段和算子 CSV。Video 阶段 CSV 同时含主体与拼接图，以 `graph_label` 区分，并保留各自 `graph_multiplier`。

默认模型运行不依赖 `measured/` 是否存在；只有实测比对脚本和捕获签名回归测试需要它。未提供实测文件时，捕获签名测试显式 skip，独立形状、次数、算术和集成测试照常运行。

当前 `A5_pr_server` 预设缺少明确的 FP32 Cube/Vector 吞吐，运行新 decode 会报错，避免将 FP16 吞吐冒充 FP32；五个未改动阶段仍可独立运行。历史源校验脚本只核对这五个阶段，不再用旧 decode 数值或旧全流程总量作为正确性标准。
