# MAGI-2 数值回归基线

本目录保存 MAGI-2 两条工作流的冻结数值，用于检查配置、算量、时延和显存统计是否发生变化。`provenance.json` 保存采集记录、文件指纹、Python/NetworkX 版本和比较容差。

| 文件 | 工作流与用途 |
| --- | --- |
| `i2v_default.json` / `t2v_default.json` | 默认 workload：1080p、10 秒、100 次 Preview、5 次 Refiner、CP=EP=8，B300 单节点 8 卡；完整工作流回归口径 |
| `i2v_minimal.json` / `t2v_minimal.json` | 同一默认配置，覆盖为 272p、Preview=1、Refiner=1；轻量场景，无 Refiner 阶段 |
| `provenance.json` | 采集环境、文件指纹及浮点比较容差 |

安装当前仓库后执行：

```bash
python scripts/magi2_baseline.py
python scripts/magi2_baseline.py --profile minimal
pytest -q tests/test_magi2_reproducibility.py
```

比较包含解析后的配置和 shape、架构指纹、逐图执行次数与时延、阶段算子数、加权 FLOPs/流量、阶段显存、CPU 工作量，以及每个算子的名称、shape、dtype、权重和工作量指纹。JSON 使用 Python 浮点的完整往返精度；浮点比较使用 `rel_tol=1e-10, abs_tol=1e-8`，整数和结构指纹精确比较。图重复次数只加权时间与工作量，不加权驻留权重和显存峰值。

`total_execution_time_ms` 是已建模加速器阶段时间；CPU 文本编码、媒体前后处理未计时，不能视为实测端到端时延。显存是解析工作集近似，容量结论为暂定；这些限制随基线一起冻结。

维护者只有在确认公式变化后才应使用 `--capture-dir <新目录>` 生成候选基线。该选项运行当前安装的实现，不会自动重写本目录；应先检查与冻结基线的差异及其原因，再决定是否建立新版本基线。
