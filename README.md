# rooflang-sim

用于多模态生成模型的解析式性能仿真。输入模型结构、工作负载、部署和硬件参数，输出算子成本、阶段时延、访存流量、解析显存和容量判断。**不加载模型权重，不生成真实视频／音频，不要求 GPU。**

目前包含 MAGI-2 Preview 的图生视频／文生视频，以及 MiniMax-H3 的首尾帧约束／参考媒体约束音视频生成。计算图由内置的离散事件模拟器执行，支持计算、访存、通信、资源竞争和多阶段统计。

## 1. 安装并运行

要求 Python 3.11+，推荐使用验证环境对应的 Python 3.12。采用标准 Python 包布局，仓库目录名称不影响安装和导入。

```bash
cd ~/Desktop/rooflang-sim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
rooflang-sim --model magi2_preview --config examples/magi2_quick.json --output-dir outputs/quick
```

基础运行仅依赖 `networkx`；测试增加 `pytest`。需要复现已验证的依赖环境时，先执行 `python -m pip install -r requirements.lock.txt`，再安装本包；锁文件包含可选绘图与测试依赖。绘制计算图时再安装 `python -m pip install -e '.[plot]'`。`python -m rooflang` 与 `rooflang-sim` 等价。

完整工作流：

```bash
rooflang-sim --model magi2_preview --hardware B300Cluster1Node --config examples/magi2_i2v.json --output-dir outputs/magi2_i2v
rooflang-sim --model magi2_preview --hardware B300Cluster1Node --config examples/magi2_t2v.json --output-dir outputs/magi2_t2v
rooflang-sim --model minimax_h3 --hardware A5_pr_server --config examples/minimax_h3_fl2va.json --output-dir outputs/h3_fl2va
rooflang-sim --model minimax_h3 --hardware A5_pr_server --config examples/minimax_h3_ref2va.json --output-dir outputs/h3_ref2va
```

命令行参数覆盖 JSON 中同名参数；其余使用模型随包提供的默认配置。`--stage <阶段名>` 只运行一个阶段，默认 `all`。`--help` 列出可用参数；完整模型参数通过 JSON 设置。

| 模型 | 工作流 | 默认硬件与部署 | 主要输入边界 |
| --- | --- | --- | --- |
| `magi2_preview` | `i2v`、`t2v` | B300 单节点 8 卡，CP=EP=8 | 10 秒，batch=1；272p／540p／1080p |
| `minimax_h3` | `fl2va`、`ref2va` | A5_pr_server，4 卡，SP=4 | 默认 124 帧、768×1344、FP16、50 次 denoise |

MiniMax-H3 的 `fl2va` 包含首帧／尾帧／首尾两帧条件；`ref2va` 支持参考图像／视频及可选音频。当前冻结的 Ref2VA profile 使用 124 帧参考视频与音轨。MAGI 的 B300/H200 预设可从 CLI 选择；MiniMax-H3 默认使用专用 A5 预设，其他硬件需明确校准 Cube／Vector 算力及通信参数。

## 2. 新模型接入前准备什么

仿真接收**结构与统计量**，不读取原始图片、视频、音频或 safetensors 权重。按下表准备可追溯数据；不要把缺失值填成零后声称分析完整。

| 数据 | 必须给出的内容 | 获取方式／用途 |
| --- | --- | --- |
| 模型结构 | 各组件的层数、hidden size、头数、head dim、MLP/MoE、卷积核／步长、VAE 压缩与 patch、模块执行顺序 | 固定版本的模型 config 与官方代码；决定计算图和 FLOPs |
| 权重与精度 | 各组件权重字节、共享／分片规则、计算和存储 dtype | 权重索引及 tensor shape；用于驻留权重和搬运，索引总量不能替代算子访存 |
| 工作负载 | batch、帧数、尺寸、音频采样数、文本 token／condition rows、参考媒体形状、窗口与 tile、扩散步数、CFG 次数 | 用实际 tokenizer／processor 先测计数；保存 processor 版本、输入描述与计数来源 |
| 部署与硬件 | TP/SP/CP/EP 等策略、卡数、各 dtype 吞吐、HBM 容量与带宽、拓扑／链路带宽／延迟／效率 | 硬件规格或标定结果；注明峰值与有效值，以及每卡／全局统计范围 |
| 验收基准 | 预期 shape、图执行次数、手算算子成本、参考计算结果、适用范围与近似 | 冻结完整精度结果和版本；模型时延与实测时延要分别标注 |

接入步骤：

1. 将架构和工作负载数据放入 `src/rooflang/programs/models/<模型>/`；配置必须校验非法值，并给出可运行的 `examples/<模型>.json`。
2. 使用 `Kernel`、`ComputeGraph`、`Placement` 搭图；显式区分计算、访存、通信和共享权重，不把聚合流量当实际存储。
3. 实现 `create_workflow(hardware, config) -> Workflow`，把阶段返回为 `StageResult`，用 `GraphResult.multiplier` 表示重复执行；显存峰值不随重复次数相乘。
4. 在 `programs/models/__init__.py` 的 `MODEL_NAMES` 中注册模型，通过统一入口导出，并增加 shape、公式、加权汇总和容量回归测试。
5. 记录建模依据、输入、运行环境、输出和局限；按照 [完整接入指南](docs/model_onboarding.md) 核对数据，参考 [可运行最小模型](examples/toy_model.py)。

```bash
python examples/toy_model.py --check --output-dir outputs/toy
```

## 3. 缺失算子如何添加

先确认缺的是成本公式还是阶段编排。通用计算核在 `src/rooflang/language/kernels/forward.py`，通信核在 `comm.py`；模型专用成本核放在模型目录。MAGI 的 `_cost/` 保存参数化成本公式，MiniMax-H3 复用适用公式并提供自己的缺失算子。

1. 从固定版本实现写出 input／weight／output shape、dtype、FLOPs、读写字节和共享规则，注明每 rank 或全局口径。
2. 继承 `Kernel`，设置实际 Tensor、`flops`／`flops_by_dtype` 和读比例；仿真访存依据放置后的 Tensor 与读比例，单改导出属性不等于改变计时。
3. 在模型搭图处添加依赖、设备和内存放置；通信使用 `CommKernel`，重复计算放进 `GraphResult.multiplier`。
4. 用小 shape 手算验证 FLOPs／bytes，再覆盖 dtype、边界、分片和共享权重；运行完整阶段回归，核对算子加权量与汇总。

完整例子、容量与流量区别、近似算子标注见 [算子扩展指南](docs/operator_extension.md)。

## 4. 输出文件及含义

统一 CLI 和 `run_and_export()` 每次生成以下报告。低层 `export_metrics()` 只生成 CSV。同模型重跑只清理它上次登记、这次已不需要的阶段 CSV，不递归删除输出目录；不同模型共用已有报告目录会报错；建议每个模型／工作流使用独立目录。

| 文件 | 内容与用途 |
| --- | --- |
| `<stage>.csv` | 此语义阶段内各图的一次执行算子明细：shape、dtype、FLOPs、访存、计算／内存／通信时间、图名和重复次数 |
| `stage_metrics.csv` | 每阶段按图执行次数加权后的工作量、阶段时延、权重、显存、未计时部分 |
| `<model>_plan_metrics.csv` | 整个已选执行计划的配置摘要、总时延、容量判定、有效性与失败原因 |
| `summary.json` | 与阶段／计划 CSV 对应的机器可读原始数值，保留嵌套结构和完整浮点精度 |
| `config.snapshot.json`、`provenance.json` | 合并后的输入、生效配置／架构、硬件快照；版本信息、运行环境和包内容 SHA-256，用于追溯与复现 |

时间单位是 **ms**，流量／存储单位是 **bytes**；原生硬件图的 GB 使用十进制。A5 预设以 GiB 表示容量、GiB/s 表示 HBM 带宽，再转换为硬件图的十进制单位；跨预设比较时需先统一单位。算子 CSV 中 `graph_multiplier` 表示该图执行次数，算子成本乘它才能与阶段工作量比较。`compute_ms`、`memory_ms`、`network_ms` 是资源工作量，不能直接相加当墙钟时延；有并行时也不能简单累加算子时延。阶段时延来自仿真时间线。`weight_read` 表示访问量，`weight_saved` 表示该行记账的权重；重复图不会把权重驻留倍增。

`timing_included=false` 或空时延表示未建模，**不表示零耗时**。`peak_activation_bytes` 是算子级代理；模型提供的 `peak_stage_memory_bytes` 是解析工作集。`is_valid=true` 表示在明确假设下容量可行，不等同于真实部署保证。只跑一个阶段时，计划汇总仅代表所选阶段。

详细字段、加权核对代码和各模型阶段表见 [输出参考](docs/output_reference.md)。快照包含额外封装字段，不能直接整个传入 `--config`；应提取 `input_config`，并保持相同包内容与硬件。

## 5. 结果复现与测试

`baselines/` 保存各工作流的固定参考结果、配置、精度与比较方法。校验脚本运行当前模型，逐项比较 shape、图执行次数、算子成本、阶段时延和显存统计。运行与基线校验只需本仓库及已安装的 Python 依赖。完整验收记录见 [validation.md](docs/validation.md)。

```bash
python -m pytest -q
python scripts/magi2_baseline.py
python scripts/minimax_h3_baseline.py
```

未安装 `.[plot]` 时跳过可选绘图测试；执行全部测试请先安装 `python -m pip install -e '.[test,plot]'`。已验证的完整测试集为 666 项通过。

| 模型／工作流 | 基线时延（ms） | 口径 |
| --- | ---: | --- |
| MAGI I2V | 48356.312568154266 | B300 8 卡，1080p，Preview=100，Refiner=5 |
| MAGI T2V | 46695.35382179358 | 同上 |
| MiniMax-H3 FL2VA | 207504.38554284864 | A5 4 卡 SP4，FP16，50 步 |
| MiniMax-H3 Ref2VA | 624059.5473882345 | 同上，参考视频＋音频 profile |

以上都是**已建模部分的预测值，不是实测端到端时延**。MAGI 的 CPU 文本编码仅计 FLOPs／内存，未标定延迟；MiniMax-H3 对 ConvTranspose、GroupNorm、FlashAttention 访存和显存峰值采用近似建模；媒体处理、阶段权重搬运等遗漏项见 [MAGI 边界](docs/magi2_preview.md) 与 [MiniMax-H3 边界](docs/minimax_h3.md)。修改公式时，应新增带版本的基线，并记录变化原因和适用范围。

## 6. 仓库结构与接口

```text
src/rooflang/
  language/                  图、Tensor、硬件、放置、计算与通信核
  runtime/                   离散事件仿真、容量、CSV、快照与追溯
  programs/models/           magi2_preview / minimax_h3
  programs/presets/          硬件预设
  programs/experiments/      Workflow 和统一运行入口
examples/                    可直接运行的配置与新模型最小示例
baselines/                   各模型的固定基线与复现数据
scripts/                     独立基线比较工具
tests/                       通用核心、模型、导出和复现测试
docs/                        接入、算子、输出与建模边界
```

仓库聚焦多模态生成的性能仿真，由通用前向算子、通信算子、图与硬件描述、离散事件运行时和模型工作流组成。通用前向核与通信优化可用于接入更多模型，计算图绘制为可选功能。

命令行入口为 `rooflang-sim`，也可使用 `python -m rooflang` 或 `python -m rooflang.programs.experiments.main`。Python API 位于 `rooflang` 命名空间，模型通过 `create_workflow(hardware, config)` 创建工作流，通过 `run_and_export()` 运行并导出结果。

模型架构与默认工作负载以 JSON 随包提供，`examples/` 中的 JSON 用于配置具体运行场景。许可证见 [LICENSE.txt](LICENSE.txt)。
