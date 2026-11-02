# MiniMax-H3 音频转置卷积通用分块模型

状态：用户已确认设计；已实现并完成本地模型与固定配置对比验证。未进行真实 NPU 验证。

## 1. 目标、范围与已作出的选择

用户选择先实现通用转置卷积分块模型，因为真实 CANN 内核和 profiling 数据无法从内网传出。禁止新增效率系数、经验惩罚、形状到时延查表或其他面向实测时延的拟合参数。

本次只替换 MiniMax-H3 音频解码中的 134 个 Conv2DTranspose 的成本模型，包括 127 个 grouped Snake UpFilter 和 7 个普通 Upsample。输入输出形状、FP32 存储、卷积语义、4590 个叶算子边界以及已有布局转换保留。普通 Conv2D、视频解码和全局效率配置不随之调整。

成功标准是计算量、访存和并行度均可从张量形状及明确的分块算法推导，并通过独立语义与调度测试；误差变小不是验收条件。不承诺通用模型能够解释 CANN 特定内核的数百倍慢路径。

## 2. Approaches

此前考虑了两条路线：实际 CANN 内核模型，以及通用分块模型。前者能按真实 tiling/指令/访存建模，但缺少无法外传的材料；用户已选择后者。全局降效率或加入实测放大倍数不满足用户约束，不作为备选。

本设计采用一个明确的通用实现：按输出位置的 stride 相位拆分、按 group 独立的 implicit-GEMM 分块。它不是对本次 CANN 内核的复刻。相位拆分跳过结构性的插零位置；输出归属唯一 tile，局部累加，避免全局 col2im 工作区与跨 tile 原子累加。不能为了增加耗时改选插零密集计算或限制空间维并行。

## 3. 几何与运算量

输出长度：

`Lout = (Lin - 1) * stride - 2 * padding + dilation * (R - 1) + 1`

Lin/Lout 是输入/输出长度，stride 是步长，padding 是两端各自的 padding，dilation 是膨胀率，R 是核长度。当前调用没有 output_padding；未来支持必须显式增加参数和测试，不能默认为现有数据猜值。

对输出位置 y 和核位置 r，只有 `y + padding - r*dilation` 可被 stride 整除，且得到的输入位置位于 `[0, Lin)` 时，才存在有效贡献。独立计算 valid_pairs（有效输入位置与核位置对数）用于语义测试。保留现有 FLOPs 输出的“input taps including cropped edges”约定，另导出有效数学 FLOPs，避免无说明地改变历史报表。

`F_reference = 2 * B * Lin * Cin * (Cout / G) * R + F_bias`

`F_valid = 2 * B * (Cin * Cout / G) * valid_pairs + F_bias`

B 是 batch，Cin/Cout 是逻辑输入/输出通道，G 是 groups；有 bias 时 `F_bias = B*Cout*Lout`，否则为零。必须验证 Cin、Cout 能被 G 整除。F_reference 保留旧口径，F_valid 是有效算术口径；两者都不是 padded 执行工作。

对于每个输出相位 q（`y % stride == q`），定义：

- Mq：该相位在 `[0,Lout)` 内的输出位置数。
- Rq：满足 `(q + padding - r*dilation) % stride == 0` 的核位置数量。
- N：每组输出通道数 `Cout/G`。
- Kq：该相位每组归约长度 `(Cin/G)*Rq`。

该相位按 Mq×N×Kq 的矩阵问题分块。空间、输出通道、batch 和 group 均允许并行；一个 tile 内串行遍历 K 块并累加。

模板采用 `Mt=16, Nt=16, Kt=8`，分别表示每块输出位置数、输出通道数和归约元素数。它们是固定的通用 FP32 分块规则，空间 16 为本设计的模板选择，通道/归约 16、8 与现有音频物理布局的块大小一致；不声称它们是 CANN 实际 tiling 或 FP32 指令形状。它们必须写入结果元数据，不能根据 measured_ms 搜索或调整。

`tasks_q = B * G * ceil(Mq/Mt) * ceil(N/Nt)`

`tile_cube_flops_q = 2 * Mt * Nt * ceil(Kq/Kt) * Kt`

`F_executed_cube = sum_q(tasks_q * tile_cube_flops_q)`

tasks_q 是相位 q 的 tile 任务数，tile_cube_flops_q 是单 tile 包含补齐的 Cube 执行工作，F_executed_cube 是所有相位真正执行的模型工作。边界越界项和 M/N/K 尾块在片上补零，不把补零算作额外的有用 FLOPs。相位 Rq=0 时不发起 Cube 工作，但仍需输出初始化或 bias 的写入。bias 按有效输出元素计 Vector 工作，不能混入 Cube。

## 4. 数据移动与存储

每个输出 tile 在片上保持累加器。一个 K panel 的输入在 tile 的 N 方向复用，权重在 M 方向复用；跨 tile 不假定缓存命中。根据上述索引映射统计每个 panel 所需的有效源元素，panel 内相同元素只读取一次。跨 panel、跨 tile 的再次加载分别计数。补零在片上产生，不从 HBM 读取不存在的元素。

HBM 字节由实际读取的 FP32 payload 与输出写入推导，不引入 cache miss、带宽惩罚或 stride 效率系数。重复读取属于流量，不能扩大常驻权重容量或峰值激活。外部 TransData 已经单列，不再重复计入内部转置成本。

边界物理张量描述和现有物理分配保持不变；算法读取有用元素的 payload 字节与分配大小分别记录。输出物理 padding 区域按一次初始化写入计流量，输出有效区域每个元素只最终写回一次。bias 在每个输出 tile 内读取一次，其向量加法按有效输出计数。输入的 output-size 元数据只按 kernel 边界读取一次。

这是 payload 层面的无跨 tile 缓存复用假设，不是硬件总线 transaction 模型。未知的 L1/L0 带宽、地址生成、实际 cache 行放大和指令固定开销不虚构成本；明确记录为本模型边界。全局 workspace 为零，片上 accumulator/panel 的逻辑容量单独导出，不作为 HBM 分配。

## 5. 并行调度与 Roofline 接入

910B3 使用现有硬件元数据中的 20 个 AI Core。每核吞吐从现有全芯片吞吐除以 core_count 得到，保留原有硬件效率，只应用一次。任务以相位为单位计算分配波次，不把 group 数直接当作活跃核心数，也不强制长序列在一核内串行。

`waves_q = ceil(tasks_q / P)`

`T_cube = sum_q(waves_q * tile_cube_flops_q / (throughput_cube / P))`

P 是核心数，waves_q 是相位 q 的波次数，throughput_cube 是硬件既有 FP32 Cube 有效吞吐（FLOP/s），T_cube 单位为秒。尾波空闲来自整数分配，不新增可调利用率。所有任务的执行 FLOPs 与关键路径波次必须分别导出；不能把空闲核时间伪装成执行 FLOPs。

Vector bias 按有效元素和相同输出 tile 归属统计；尾 tile 的实际元素量不同，使用各波最忙核的工作量，不凭空填满 Vector FLOPs。相位按固定顺序处理，当前模板不对相位之间重排优化。

继续使用现有 kernel 级 Roofline：计算时间是 Cube 与 Vector 所需时间之和，访存时间来自模型 payload 除以已有 HBM 有效带宽，kernel 时间取二者最大值。计算波次是串行相位的调度近似，内存与计算仍采用 kernel 总量的理想重叠；本次不引入流水线启动或同步常数，不将其描述为逐周期仿真。

在 H3 专用 operator/kernel 中保存结构化 schedule，给 runtime 增加最小的可选计算时间调度接口；普通 kernel 仍走原有路径。禁止把尾波等价成本灌入现有 FLOPs 字段。CSV 的 compute_ms 必须使用与模拟同一入口计算，避免展示和计时分叉。

当前只有 910B3 的 core_count 证据完整。本次新计时路径先限定 Ascend910B3 的 H3 音频转置卷积；其他硬件保留原路径并明确标识未应用本模型，不猜 A5 核心数。

## 6. 实现位置与导出

新增 H3 专用的转置卷积成本模块，`audio_decode.py` 仅在 transpose 分支使用；`decode_ops.py` 的现有普通算子行为不变。桥接层传递 hardware/core_count，并保持既有叶算子边界和物理形状。

每个转置卷积至少可审计：reference FLOPs、valid FLOPs、executed Cube/Vector FLOPs、M/N/K 分块规则、相位任务数、核心数、波次数、重复输入/权重读取、输出写入、边界物理字节、片上 buffer 大小、全局 workspace 和模型适用范围。字段可用规范 JSON 聚合，避免无必要的大量 CSV 列。

元数据应说明 `generic_phase_tiled_transpose`、`not_cann_kernel_reconstruction`、`uncalibrated`，不继续把新成本称为实测内核算法。输入输出拓扑仍来自已有 trace。

## 7. 验证与对比

1. 用独立的小尺寸标量转置卷积枚举，验证相位覆盖、裁剪、padding、dilation、groups、输出和有效贡献计数，不只重写相同公式作为测试。
2. 对齐/尾块、任务数小于/等于/大于核心数、零贡献相位、带 bias 等场景，手工核对执行工作和波次；检查无遗漏、无重复输出及 memory/capacity 分离。
3. 保留 4590 节点和原 capture 的形状、dtype、调用次数回归；检查非 transpose 算子、视频模型、普通 runtime kernel 的数值不变。比较无 measured 目录的运行，证明预测不依赖实测时延。
4. 使用报告原配置（计算效率 0.65、HBM 0.60、其他条件固定）生成独立的新音频结果；当前 preset 默认 0.70 不改动。逐形状报告改变了多少执行工作、流量和时间，保留剩余误差。旧 reports、measured 和原实验目录不覆盖。
5. 若分块模型仍显著低估，结论是通用模板未解释实际内核，而不是继续调整 tile 或效率直到命中实测。性能数据只用于结果评估。

## 8. 留给实现者的选择

类和字段命名、是否缓存重复形状、如何用闭式求和或周期计数避免逐元素枚举、局部测试文件组织由实现者决定。生产成本计算不能按数十万个输出元素建立 Python 对象或把每个 tile 展开成全局 DES kernel；保留一个实际叶 kernel 的边界。

用户已在审阅本设计后确认实施；固定 tile、跨 tile 缓存假设和初期硬件范围按本设计落地，未引入拟合参数。

## 9. 设计核对

独立 builder check 未发现阻碍实施计划的重要问题。纯形状计算也确认：Snake 的 `(C,Lin)=(8,165610)、(16,82810)、(32,41410)` 三组在本模板中具有接近的任务数和执行 FLOPs。这是允许空间维并行的通用算法自然结果；不会为重现实测耗时随长度显著增长而添加串行化假设。代码实现后仍需运行第 7 节的独立测试，设计核对不替代测试。
