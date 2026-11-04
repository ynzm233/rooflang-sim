# 模型效率配置

效率系数由每次仿真的模型 JSON 提供，H3、MAGI 和新模型使用同一接口。硬件预设保存物理峰值与拓扑；全局 HBM `memory_efficiency` 是此次迁移保留的例外，B3 仍为 `1.0`，A5 仍为 `0.8`。它会影响计算和 DMA 的 HBM 带宽，不与单元访存效率混为一项。

## 配置与默认值

在原有工作负载 JSON 中增加可选的 `efficiency` 对象，例如：

```json
{
  "efficiency": {
    "cube_efficiency": 0.8,
    "vector_efficiency": 0.6,
    "cube_memory_efficiency": 0.8,
    "vector_memory_efficiency": 0.6,
    "intra_node_efficiency": 1.0,
    "collective_efficiencies": {}
  }
}
```

上述片段需要合入模型工作负载，不包含独立运行所需的所有形状参数。完整配置见 [H3 Ref2VA](../examples/minimax_h3_ref2va_910b_sp1.json) 和 [MAGI quick](../examples/magi2_quick.json)。CLI 仍使用 `--config`，没有额外的效率命令行参数。

| 字段 | 作用 | 缺省值 |
| --- | --- | --- |
| `cube_efficiency` | Cube/矩阵计算吞吐系数 | `1.0` |
| `vector_efficiency` | Vector 计算吞吐系数 | `1.0` |
| `cube_memory_efficiency` | Cube 所属流量的本地访存效率 | `1.0` |
| `vector_memory_efficiency` | Vector 所属流量的本地访存效率 | `1.0` |
| `intra_node_efficiency` | 同节点通信的带宽系数 | `1.0` |
| `collective_efficiencies` | aggregate 通信按参与设备数查表的附加带宽系数 | `{}`，即统一 `1.0` |

省略整个对象或仅填写部分字段时，其他系数均为 `1.0`。显式标量和表内数值必须为有限数且满足 `0 < η ≤ 1`；布尔值、数值字符串、未知字段及非法卡数键会报错。通信表的键为正整数，JSON 中写作 `"2"`、`"4"` 等字符串。空表与省略表等价。

参数只对本次选择的硬件生效。换硬件时不会自动换系数，也不会从旧硬件预设补回历史值。每次创建工作流使用独立的硬件副本，复用同一原始硬件对象不会累乘前一次运行的效率。

## 计算、访存与通信口径

H3 保留已有 dtype 与 Cube/Vector 专用峰值；MAGI 保留原来按 dtype 的统一计算峰值基准，仅分别应用两个单元的效率，不在此重构中更换其向量峰值假设。效率只调整时间；数学 FLOPs、张量流量、权重和容量定义不变。

H3 的 `transformer_fia_compute_efficiency` 保持原位置和含义：只在 B3 的 `transformer/transformer_ref` 原始 FIA 路径中，从物理峰值计算并**替换** Cube/Vector 效率，不再叠乘通用系数。它不覆盖 Qwen FIA、text refiner、decode、访存或 DMA。

本地访存使用已有单元流量拆分；若字节数为 `B_C/B_V`，效率为 `η_C/η_V`，则有效效率为：

```text
η_effective = (B_C + B_V) / (B_C / η_C + B_V / η_V)
```

零流量取 `1.0`。没有单元流量拆分的混合算子整体归 Cube；纯 Vector、storage 和无计算分量的搬运算子归 Vector。不能根据 FLOPs 比例拆分访存字节。DMA WeightLoad 维持自己的传输路径，不套用计算核的 Cube/Vector 访存系数。

通信带宽系数只影响字节传输时间，不改变固定启动延迟。原生点对点 AllToAll/Send/Recv 沿用点对点路径，应用同节点系数；aggregate 路径另乘 `collective_efficiencies`。卡数取当前通信核实际参与的去重设备数。查表选绝对距离最近的键，等距时选配置中先出现的键；例如按 `2, 4, 8` 顺序给表时，`3→2`、`6→4`。不插值，也不把未列出的卡数强设为 `1.0`。

`intra_node_efficiency` 仅作用于参与设备全部处于同一节点的通信；跨节点 aggregate 保留原有带宽公式，仍可使用按参与卡数定义的 collective 表。现有 GPU 预设按其节点分组，950DT 按 machine 分组；新硬件可在计算节点声明一致的 `node_id`。没有分节点信息的普通硬件图按单节点处理。

## 从旧配置迁移

仓库已有示例已显式记录原有效系数，因此原示例无需额外命令参数。B3 示例使用 Cube 算力/访存 `0.8`、Vector 算力/访存 `0.6`；MAGI 的 B300 示例全部为 `1.0`。

复现旧 H3 A5 场景时，模型配置填写：

```json
{
  "efficiency": {
    "cube_efficiency": 0.7,
    "vector_efficiency": 0.6,
    "intra_node_efficiency": 0.7,
    "collective_efficiencies": {"2": 0.142857, "4": 0.428571, "8": 1.0}
  }
}
```

A5 两项单元访存系数原来未单独启用，仍取 `1.0`；全局 HBM 的 `0.8` 留在硬件中。旧 MAGI 若使用 NPU，其按 dtype 的统一计算基准原先实际采用 Cube 系数，复现时 Cube/Vector 两项均需填写这一原有效系数，不应误以为旧 MAGI 使用了 NPU 专用 Vector 峰值。

自定义旧配置没有填写 `efficiency` 时，会按新默认值运行，结果可能变化。不要通过修改硬件常量继续设置这些系数；应修改本次模型配置。历史结果、测量文件和既有快照不改写。

## 新模型接入与审计

公共实现为 [`runtime/efficiency.py`](../src/rooflang/runtime/efficiency.py)。新模型的公开 Python 入口和 CLI 必须走相同路径：

1. 使用 `resolve_efficiency(config.get("efficiency"))` 校验并补齐系数，将结果保存到模型的 resolved config。
2. 在首次建图或内部估时前调用 `prepare_hardware(hardware, efficiency)`，后续图、placement、内部估时和仿真均使用返回的硬件副本。
3. 计算核保留 Cube/Vector 工作量及对应峰值口径。有 `operator` 成本接口时调用 `apply_memory_efficiency(kernel, hardware)`；原生自定义核显式提供 `dtype:cube` / `dtype:vector` 计时键与 `local_memory_efficiency`，参考 [toy_model.py](../examples/toy_model.py)。
4. 返回 `Workflow(..., hardware=hardware)`，通过 `run_and_export()` 导出。Snapshot 在运行前记录该副本，`input_config` 保留原输入，resolved config 与 `hardware.metadata.efficiency` 记录实际系数。节点吞吐为生效值，通信边保留物理带宽，通信计时另用效率系数。
5. 覆盖默认/部分填写/非法值、单元效率改变时间但不改变字节数、连续运行隔离、快照一致性，以及相同有效系数下的迁移前后对照。

低层直接构造 `Compute`、`FabricEdge`、`Simulator` 的调用仍使用调用者提供的吞吐值；没有模型接入步骤时不会自动读取某个模型配置。效率数值是解析仿真假设，不代表真实硬件测量或完成校准。
