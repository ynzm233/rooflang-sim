# 交付验收（2026-10-15）

本记录针对独立 `rooflang-sim`，不是对真实设备性能精度的承诺。

## 已执行的检查

| 检查 | 结果 |
| --- | --- |
| 新建 Python 3.12.13 虚拟环境，标准 editable 安装 | 成功；基础依赖仅 NetworkX，测试／绘图是可选依赖 |
| `python -m pytest -q` | 666 项通过，25.46 秒；覆盖通用核心、算子、通信、容量、阶段、导出和四工作流回归 |
| `python scripts/magi2_baseline.py` | I2V/T2V 默认完整工作流通过；另有 272p 轻量配置回归 |
| `python scripts/minimax_h3_baseline.py` | FL2VA 4288 行、Ref2VA 4282 行逐算子与阶段汇总通过 |
| `python examples/toy_model.py --check` | 3 项通过；独立手算 FLOPs／流量／驻留和重复统计 |
| 构建 wheel，安装到只含 NetworkX 的另一新环境 | 成功；不使用 editable 路径 |
| 安装后的 wheel 在审计钩子下跑四条完整工作流 | 全部通过；禁止网络和两个源仓库／构建目录读取，仅允许安装包、Python 运行时和输出目录 |
| 桌面最终目录使用独立系统 Python 3.14.2，最小 test 安装 | 645 项通过、1 个可选绘图模块跳过，9.74 秒；不依赖旧库内的 Python 运行时 |
| 系统 Python 3.14.2 的非 editable wheel 离线审计 | 四条完整工作流再次通过，旧仓库及构建目录的任何读取均被禁止 |
| 文档相对链接、Python 语法 | 通过 |

验收环境的固定依赖见 `requirements.lock.txt`。包内 `.py/.json` 内容 SHA-256：

```text
c5e2944564941d39a90c0e73e1dcfc88b8d0b2feabc4a40e8cc46852bc2f817a
```

该哈希与安装后的 wheel 一致；每次运行的 `provenance.json` 会重新计算，可发现公式或默认配置漂移。

## 冻结基线

| 工作流 | 源时间 ms | 迁移时间 ms |
| --- | ---: | ---: |
| MAGI I2V | 48356.312568154266 | 48356.312568154266 |
| MAGI T2V | 46695.35382179358 | 46695.35382179358 |
| MiniMax-H3 FL2VA | 207504.38554284864 | 207504.38554284812 |
| MiniMax-H3 Ref2VA | 624059.5473882345 | 624059.5473882316 |

浮点累加按声明容差比较，不通过重写基线来消除差异。shape、图执行次数和身份结构按精确值核验。H3 的源 YAML、完整原始 CSV 与代码指纹保存在 `baselines/minimax_h3/`；原库缺少 safetensors index 原件的事实保留在文档中。

源版本：RoofLang `695b239b0fd349ee08a705578893ab8fe4de1d8b`，model_Sim `f27d4bdfa823fe0b151749cc5d1e1d734e7fb583`。工作期间未修改两份源代码；model_Sim 原有的 MAGI 输出 XLSX 修改保持原样。

## 审查中修复的问题

- 单阶段 H3 也包含该阶段 indexed 权重、持久张量和 activation proxy；不再以图权重替代完整阶段容量，也不声称整流程可行。
- JSON 序列化失败先于任何报告发布，避免新 CSV 与旧配置快照混在一起；CLI 拒绝 NaN／Infinity。不同模型不能混用一个已有报告目录。
- 新包使用标准 `src` 布局，移除父目录 `.pth`、固定仓库名、训练／反向／优化器和纯 LLM 专用接口。

时延仍是已建模主算子的估计，显存仍是明确标注的解析代理；MAGI CPU 未计时部分、H3 原有近似和硬件单位兼容口径详见两份模型说明。
