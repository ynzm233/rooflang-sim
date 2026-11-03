# 当前模型验证

核对日期：2026-11-03。以下结果来自当前仓库源码及本地 macOS `.venv`：Python 3.14.2、NetworkX 3.7。命令给出 Windows PowerShell 等价写法；没有据此声称已完成 Windows 或真实 NPU 验收。

## 已执行检查

| 检查 | 本次结果 |
|---|---|
| `.\.venv\Scripts\python.exe -m pytest -q` | 707 passed，1 skipped；跳过可选绘图模块；12.29 秒 |
| `.\.venv\Scripts\python.exe scripts/magi2_baseline.py` | I2V／T2V 默认配置均通过数值回归 |
| MiniMax-H3 Ref2VA 单卡完整 CLI | 七阶段完成，`is_valid=true`，按随附主配置和当前 B3 预设运行 |
| `.\.venv\Scripts\python.exe scripts/minimax_h3_decode_validation.py --output-dir outputs/minimax_h3_decode_validation` | Video／Audio 调用数分别为 459714／4590，与实测明细一致；误差详见 decode 文档 |
| `.\.venv\Scripts\python.exe examples/toy_model.py --check` | 3 项独立形状、非法配置与重复记账检查通过 |

本次运行记录的包内 `.py/.json` 内容 SHA-256：

```text
1f792110bba4d65327a186f8bbb736a430f99684393684a08af33c241895e551
```

每次运行由 `provenance()` 计算并导出哈希；它覆盖包内代码和配置，不包含本文。依赖约束见 `pyproject.toml` 和 `requirements.lock.txt`。

## 当前运行结果

| 工作流 | 已建模时延 ms | 场景 |
|---|---:|---|
| MAGI I2V | 48356.312568154 | B300 8 卡、1080p、100 次 Preview／5 次 Refiner |
| MAGI T2V | 46695.353821794 | 同上 |
| MiniMax-H3 Ref2VA | 5447582.931051 | Ascend910B3 单卡、SP1、49 步、组件混合精度、逐层卸载 |

H3 使用 `examples/minimax_h3_ref2va_910b_sp1.json`：Cube 计算／访存效率 0.8，Vector 计算／访存效率 0.6，未设置联合 DiT FIA 效率覆盖。每卡容量代理峰值为 13187512628 bytes；容量通过不等于真实框架可部署。阶段数值及完整命令见 [MiniMax-H3](minimax_h3.md#当前-ref2va-结果)。

## 验证边界

算术、图依赖、精度、调用次数、容量记账和导出一致性由本地测试验证；这些结果不能证明真实 kernel 时延准确。decode 比对关闭逐层卸载，使用设备 kernel 时长之和，与完整流程的含搬运时间不同。

当前 Video decode 预测 17458.638510 ms，相对实测 kernel 总时间误差 -14.25%；Audio decode 预测 95.581845 ms，误差 -89.92%。不设通过阈值，音频时延仍显著低估。host_total、device_span 与 kernel_total 分别解释，不能相加或混为同一口径，详见 [decode 建模说明](minimax_h3_decode.md)。

`A5_pr_server` 没有 decode 所需的 FP32 吞吐，不作为当前完整 H3 流程的运行预设。MAGI CPU、媒体前后处理及真实内存分配器等未计时／未建模部分，见各模型说明与 [输出参考](output_reference.md)。
