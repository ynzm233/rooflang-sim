# MiniMax-H3 历史源基线

本目录的 `fl2va/`、`ref2va/`、`source/` 和 `provenance.json` 是源工程
`model_Sim` 提交 `f27d4bdfa823fe0b151749cc5d1e1d734e7fb583` 的冻结复现证据。
`source/` 内的 YAML 和记录的 SHA-256 保持原样，不作为当前运行配置加载。

2026-10-15，生产 `A5_pr_server` 已改用官方 950PR 规格和十进制内存单位，
保留必要的效率、通信仿真假设。当前有效配置见
[`hardware.py`](../../src/rooflang/programs/models/minimax_h3/hardware.py)，
参数依据见 [硬件规格与仿真假设](../../docs/minimax_h3.md#硬件规格与仿真假设)，
当前运行结果见 [模型文档](../../docs/minimax_h3.md#运行与复现)。

`python scripts/minimax_h3_baseline.py` 使用脚本内隔离的历史硬件重放本目录，
校验模型算子和工作量与源工程一致。该历史硬件不注册到日常 CLI；
新硬件的计算、访存、容量和通信验证见 `tests/test_a5_950pr.py`。
