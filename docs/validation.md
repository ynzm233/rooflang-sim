# 交付验收（2026-10-15）

下表命令已换为 Windows PowerShell 等价写法；测试结果和耗时保留原验收记录，不代表已在 Windows 重新测试。

本记录对应当前 `rooflang-sim`：`A5_pr_server` 已采用官方 Ascend 950PR 规格，并明确保留必要的效率与通信仿真假设。运行环境为仓库 `.venv`，Python 3.14.2、NetworkX 3.7。

## 已执行的检查

| 检查 | 结果 |
| --- | --- |
| `.\.venv\Scripts\python.exe -m pytest -q` | 652 项通过，1 个可选绘图模块跳过，9.47 秒；覆盖核心、算子、通信、容量、阶段、导出和模型回归 |
| `tests/test_a5_950pr.py` | 7 项通过：Cube/Vector 手算成本、FP8 不支持路径、十进制带宽、128 GB 容量边界、通信假设与快照 |
| `.\.venv\Scripts\python.exe scripts/minimax_h3_baseline.py` | 历史硬件源复现通过：FL2VA 4288 行、Ref2VA 4282 行；6 份源配置 SHA-256 校验通过 |
| 当前默认 CLI 两条 H3 完整工作流 | FL2VA、Ref2VA 均完成；默认阶段卸载下两者 `is_valid=true` |
| 当前 CLI 导出 | `config.snapshot.json` 包含官方规格、来源、核查日期与仿真假设；容量和带宽采用十进制 |

依赖约束见 `pyproject.toml` 和 `requirements.lock.txt`。当前包内 `.py/.json` 内容 SHA-256：

```text
ce86c4a70d8ca8d5166b211928362e27a19682aaeb35a450f9c2644ec2010b92
```

该哈希由当前源码的 `provenance()` 计算；每次运行的 `provenance.json` 会重新计算，用于发现公式或默认配置漂移。

## 当前默认运行结果

| 工作流 | 主算子时延 ms | 硬件与工作负载 |
| --- | ---: | --- |
| MAGI I2V | 48356.312568154266 | B300 8 卡，1080p，Preview=100，Refiner=5 |
| MAGI T2V | 46695.35382179358 | 同上 |
| MiniMax-H3 FL2VA | 228193.52842317586 | 官方 950PR 规格＋仿真假设，4 卡 SP4，FP16，50 步 |
| MiniMax-H3 Ref2VA | 685813.7461188152 | 同上，参考视频＋音频 profile |

H3 当前结果由修正后的原生仿真得到。`baselines/minimax_h3/` 中的源工程快照和逐算子 CSV 用于隔离的历史硬件复现，不作为当前默认参数的性能结果。浮点累加按声明容差比较，shape、图执行次数和身份结构按精确值核验。权重字节数来自静态配置；仓库不包含对应的 safetensors index 原件，未完成对权重索引的独立核验。

## 审查中修复的问题

- 单阶段 H3 也包含该阶段 indexed 权重、持久张量和 activation proxy；不再以图权重替代完整阶段容量，也不声称整流程可行。
- JSON 序列化失败先于任何报告发布，避免新 CSV 与旧配置快照混在一起；CLI 拒绝 NaN／Infinity。不同模型不能混用一个已有报告目录。
- 包使用标准 `src` 布局，可安装到独立虚拟环境并从任意工作目录运行。

时延是已建模主算子的估计，显存是明确标注的解析代理；MAGI CPU 未计时部分、H3 近似项和硬件参数单位详见两份模型说明。

## A5 参数口径

当前 `A5_pr_server` 采用 32 Cube／64 Vector 的 950PR 规格：Cube FP16／FP8 为
432／865 TFLOPS，Vector FP16 为 54 TFLOPS，内存为十进制 128 GB／1600 GB/s。
有效吞吐继续应用明确标注的仿真效率。删除未使用的 Vector FP8 数值，通信和部署限制
保留为仿真假设，官方规格和假设分别随硬件快照导出。来源链接见
[硬件规格与仿真假设](minimax_h3.md#硬件规格与仿真假设)。

当前时延是修正参数后的原生仿真结果，不是重跑上游得到的新源基线，也不是实测性能。
