# 接入新模型：数据准备与可运行示例

先运行仓库自带的接入示例，再替换其形状、算子和阶段。此仓库模拟计算、访存、通信和内存占用；输入是模型描述与数值工作负载，不需要执行模型前向或加载权重。

```bash
python -m pip install -e .
python examples/toy_model.py --check
python examples/toy_model.py --tokens 32 --hidden 16 --steps 3 --output-dir outputs/toy
```

`--check` 使用 Python 标准库运行三个检查，无需另装 pytest。正常运行生成 `transform.csv`、`stage_metrics.csv`、`toy_affine_plan_metrics.csv` 和三个运行记录 JSON。示例设备为自行声明的 1 BF16 TFLOP/s、100 GB/s、1 GB 内存设备，不代表真实产品。

## 1. 准备能追溯来源的输入

将以下五类信息整理成 JSON、表格或建模说明。JSON 的具体字段由模型自己的 `config.py` 校验；不能把其他模型的配置直接改名后视为已经接入。

| 输入类别 | 必须记录的内容 | 获取方式与检查 |
| --- | --- | --- |
| 架构与执行流程 | 组件、层数、hidden/head 维度、FFN、专家数/top-k、注意力类型、VAE/patch 压缩倍率、阶段依赖、CFG 分支、去噪次数 | 固定官方配置/代码版本；记录一个输入如何经过各组件，列出未覆盖组件 |
| 权重与状态存储 | 参数形状、存储 dtype、量化 scale/zero-point 字节、共享权重、每个组件常驻字节、KV/其他持久状态 | 可离线读取 checkpoint 索引或张量元数据；单独保存提取后的数值及文件哈希。只有 `weight_map` 的索引不一定含张量尺寸，不能直接当作逐组件字节清单 |
| 处理器输出与模态尺寸 | 正/负文本的实际 token 数、特殊 token、padding/mask；图像尺寸/裁剪；视频帧数/fps/时长；音频采样率/采样数；编码后 latent、patch/token 长度 | 用对应版本 tokenizer/processor 离线测得，或采用有来源的固定样本。记录 token 数位于 tokenization、padding、CP 分片中的哪一步 |
| 部署与通信 | batch/concurrency、TP/CP/EP/DP/PP、各组件设备、每卡权重与工作量、通信算子/参与者/payload/dtype、加载卸载顺序 | 从实际部署方案或明确假设得到；说明单位是全局、单卡、临界 rank 还是全部参与者合计 |
| 硬件与近似 | 每种计算 dtype 的 TFLOP/s、内存容量 GB、读写 GB/s、互连拓扑/方向/启动延迟 μs；未标定项、校准倍率、来源版本与哈希 | 优先采用可追溯的规格或测量；标明十进制 GB 与二进制 GiB。区分理论峰值、有效吞吐与实测延迟，未知项不要填成 0 |

原始图片、音频、视频和完整权重通常不进入仿真输入目录。保留由它们导出的数值描述即可复现解析模型；若某个值依赖样本内容，例如稀疏率、路由分布或压缩后 token 数，需要一并固定。

可按下面结构保存自己的输入材料；这是数据组织建议，不是所有模型通用的配置 schema：

```text
new_model/
├── architecture.json       # 组件、层、维度与算子参数
├── workload.json           # 一次请求的尺寸、阶段和迭代数
├── parameter_bytes.json    # 逐组件/共享组的权重与状态字节
├── deployment.json         # 放置、并行划分与通信
└── SOURCES.json            # 每份数据的来源、版本、哈希与近似
```

在此基础上，先手算一个很小的样本：每个阶段的形状、FLOPs、流量、驻留权重与调用次数。它是验证程序的独立依据，不从待测程序生成。

## 2. 从可运行示例建立计算图

完整实现见 [`examples/toy_model.py`](../examples/toy_model.py)。示例执行 `y = x * scale + bias`，输入/输出为 BF16 的 `[batch_size, tokens, hidden]`，两个权重为 `[hidden]`。

令 `N = batch_size × tokens × hidden`，一次图调用的公式为：

```text
FLOPs       = 2 × N                         # 每元素一次乘、一次加
输入流量    = 2 × N 字节
输出流量    = 2 × N 字节
权重流量    = 驻留权重 = 4 × hidden 字节
总访存流量  = 4 × N + 4 × hidden 字节
```

例如 `batch_size=1, tokens=2, hidden=4` 得到 16 FLOPs、48 字节总访存、16 字节权重。示例硬件无启动开销，单次耗时为 `max(16 / 10^12, 48 / 10^11)` 秒。`--check` 用这个独立结果验证仿真器输出。

也可把以下对象保存为任意 JSON 文件，例如 `toy_workload.json`：

```json
{"batch_size": 1, "tokens": 2, "hidden": 4, "steps": 3}
```

```bash
python examples/toy_model.py --config toy_workload.json --output-dir outputs/toy_small
```

执行链为 `create_workflow(hardware, config)` → 构造 `Kernel`/`Tensor` → 加入 `ComputeGraph` → `Placement` 指定设备和内存 → 校验 → `simulate()` → `StageResult` → `run_and_export()`。示例通过 Python 接口运行，不占用正式模型注册表中的名字。

## 3. 把真实模型接到相同接口

1. 在 `src/rooflang/programs/models/new_model/` 实现配置读取和校验。未知字段、非整数/负数维度、不支持的工作流、不能整除的分片和设备不足应显式报错。推导并保存解析后的形状，避免在多个模块各自计算。
2. 按阶段搭建计算图。优先复用 `src/rooflang/language/kernels/`；缺少的公式按[算子扩展说明](operator_extension.md)实现。数据边描述真实张量依赖，控制边只约束顺序。并行部署需要显式表示每 rank 的工作量及通信，不能将所有全局统计简单除以卡数。
3. 提供 `create_workflow(hardware, config) -> Workflow`，并在模型目录的 `__init__.py` 导出。阶段回调返回 `SimulationResult` 或 `StageResult`。通过 metadata 说明生效配置、统计范围、请求计数与未建模工作；跨阶段权重驻留/卸载与状态延续需要显式处理。
4. 在 `src/rooflang/programs/models/__init__.py` 注册模型名。正式 CLI 通过注册表加载工作流；新模型参数优先放入 `--config`，无需为每个参数增加 CLI 选项。模型专属 CSV 字段在 `src/rooflang/runtime/export_config.py` 的 `MODEL_EXTENSIONS` 声明，模型提供取值。
5. 加入小样本解析测试、配置拒绝测试和一次完整 CLI 测试；确认结果不读取外部源仓库、权重或历史输出文件。再运行正式规模案例，并把配置、来源记录和可比较的参考结果一起提交。

注册后的命令形式为：

```bash
rooflang-sim --model new_model --hardware B300Cluster1Node \
  --config examples/new_model.json --output-dir outputs/new_model
```

这条命令只有在完成注册并创建相应配置后才可执行；本仓库随附的 `toy_model.py` 已可直接运行。

重复开销完全一致的图可表示为：

```python
from rooflang.runtime.stages import GraphResult, StageResult

stage = StageResult([GraphResult("denoise_step", result, multiplier=20)])
```

若每步的形状、KV 长度、稀疏率或通信发生变化，应分别仿真相应图。`multiplier` 放大计算量、流量和耗时，不放大驻留权重或内存峰值。阶段回调默认顺序执行；多个独立阶段的时间线不会自动重叠。

## 4. 复现与验收

| 检查 | 通过条件 |
| --- | --- |
| 数学与形状 | 独立小样本的 FLOPs、dtype、字节数、取整/padding 和分片边界一致 |
| 工作流 | 条件分支、阶段集合、CFG 倍数、窗口数和去噪次数与模型声明一致 |
| 内存与通信 | 流量和存储分开；共享权重去重；通信参与者/方向/消息量正确；容量判定有明确范围 |
| 输出与复现 | 相同源码、依赖、硬件及生效配置重复运行得出相同指标；比较数值时声明浮点容差 |
| 独立运行 | 安装后可从不同工作目录运行，不依赖原 `rooflang` 或 `model_Sim` 路径，不把参考 CSV 当成结果生成输入 |

保存一组完整输出的做法见[输出文件说明](output_reference.md)。修改公式、默认值或硬件参数后，应更新来源记录并重跑参考案例；不能只更新预期数字来消除测试失败。

## 5. 当前最小集合与 MAGI 范围

本仓库保留通用 DSL、仿真器、算子、硬件与导出，以及 MAGI-2 Preview 和 MiniMax-H3 建模。纯 LLM 案例和专用于它们的搜索入口不属于本次多模态最小集合。公共算子虽然可能同时用于 LLM，仍是新模型接入和内部图校验需要的基础设施。

MAGI 的本地模型描述、公式和来源记录位于 `src/rooflang/programs/models/magi2_preview/`。当前支持 I2V/T2V、272p/540p/1080p、10 秒视频、batch=1、CP=EP∈{1,2,4,8}；540p/272p 没有精修阶段。CPU 文本前向仅统计算量和内存，未标定延迟；冷加载、后处理和封装未计时。其吞吐量是已建模加速器阶段的片段生成率，显存为解析工作集估计。详细口径见 [`magi2_preview.md`](magi2_preview.md)。
