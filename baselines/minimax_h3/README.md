# MiniMax-H3 回归数据

本目录供 `scripts/minimax_h3_baseline.py` 执行限定范围的源一致性检查，覆盖 Qwen vision、Qwen text、条件 VAE、text refiner 与联合 DiT 五个阶段。脚本使用独立的验证硬件参数；本目录数据不作为当前 CLI 的输入或当前 B3 性能结果。

当前模型的完整流程验证使用 [Ref2VA 主配置](../../examples/minimax_h3_ref2va_910b_sp1.json) 与 Ascend910B3；decode 使用专门的结构和成本验证脚本：

```powershell
.\.venv\Scripts\python.exe -m rooflang --model minimax_h3 --hardware Ascend910B3_1Card `
  --config examples/minimax_h3_ref2va_910b_sp1.json --output-dir outputs/h3-ref2va
.\.venv\Scripts\python.exe scripts/minimax_h3_decode_validation.py `
  --measured-dir measured --output-dir outputs/minimax_h3_decode_validation
```

算子身份、shape 和调用次数精确核对；数值回归浮点容差为 `rel_tol=1e-10, abs_tol=1e-8`。当前公式、硬件假设和时延见 [模型说明](../../docs/minimax_h3.md)，复现命令见 [结果复现与测试](../../README.md#5-结果复现与测试)。
