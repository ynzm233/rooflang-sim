# 缺少算子时如何添加

先确定算子的输入/输出形状、计算精度、FLOPs、读写流量和驻留存储，再实现 `Kernel`。可直接运行的完整示例是 [`examples/toy_model.py`](../examples/toy_model.py) 中的 `Affine`。

```bash
python examples/toy_model.py --check
```

## 1. 选择扩展位置

| 情况 | 位置与做法 |
| --- | --- |
| 已有等价的矩阵乘、注意力、归一化、卷积等算子 | 先检查 `src/rooflang/language/kernels/`，复用并核对其公式、复用假设和精度语义 |
| 只被一个模型使用的计算核 | 放入 `src/rooflang/programs/models/<model>/operators.py`，直接继承 `Kernel` |
| 多个模型共享、语义和口径一致的计算核 | 放入 `src/rooflang/language/kernels/` 对应模块，并添加公共测试 |
| 通信操作 | 优先复用 `language/kernels/comm.py`；新操作继承 `CommKernel` 并明确参与者、端口数、payload 和算法 |
| 迁移来源已有分阶段/复合公式 | 可保留模型内部公式模块，再显式转换成原生 `Kernel`；参考 MAGI 的 `_cost/` 与 `bridge.py`，无需给每个新模型复制整套公式框架 |

Python 算子类不需要加入一个全局算子注册表才能仿真。模型图构建函数实例化并加入 `ComputeGraph` 即可。MAGI `_cost` 的注册表属于其迁移适配层，不是新模型的强制接口。

## 2. 先写算子契约与手算案例

在 docstring 或建模说明中写出以下五项：

1. 输入、权重、输出的形状及 dtype，支持的维度范围、padding 和分片限制。
2. FLOPs 公式与计数约定。例如矩阵乘 `M×K` 乘 `K×N` 按 `2MKN`；FMA 算两次，exp/sqrt 等操作的等效 FLOP 约定需要注明。
3. 每次调用读多少输入、读多少权重、写多少输出；说明缓存复用、稀疏读取、量化附加数据和重复访问。
4. 真正同时驻留的权重、状态、输入/输出和临时工作区；它们通常不等于累计流量。
5. 计算与访存是否重叠、哪些子步骤串行、采用何种性能近似、公式来源及独立小样本的预期值。

以 BF16 仿射变换 `y = x * scale + bias` 为例，`x/y` 为 `[B,T,D]`，权重均为 `[D]`。令 `N=BTD`：`FLOPs=2N`，输入/输出合计 `4N` 字节，两组权重 `4D` 字节。对 `[1,2,4]`，应得到 16 FLOPs、48 字节访存和 16 字节驻留权重。

## 3. 实现形状与开销

下面的类可以直接放入模型的 `operators.py`。完整示例还包含输入校验、硬件、工作流、命令行和检查。

```python
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.tensor import Tensor


class Affine(Kernel):
    def __init__(self, batch_size, tokens, hidden):
        if any(type(n) is not int or n <= 0 for n in (batch_size, tokens, hidden)):
            raise ValueError("dimensions must be positive integers")
        shape = (batch_size, tokens, hidden)
        super().__init__(
            inputs={"x": Tensor("bf16", shape)},
            weights={
                "scale": Tensor("bf16", (hidden,), weight_id="affine.scale"),
                "bias": Tensor("bf16", (hidden,), weight_id="affine.bias"),
            },
            outputs={"y": Tensor("bf16", shape)},
        )

    @property
    def flops(self):
        return 2 * self.inputs["x"].n_elements
```

基础类默认根据张量计算输入、权重和输出字节。`weight_id` 用于标识共享的物理权重；同一份权重重复引用应使用相同身份，不同层的独立权重必须使用不同身份。上述固定 ID 只适用于示例中的一个仿射模块，复制到多层模型时应加入层/模块标识。

**计时和导出必须使用相同流量。** 仿真器根据已放置的张量、`input_read_fraction(port)` 和 `weight_read_fraction` 计算访存时间。只重写 `input_bytes`、`output_bytes` 或 `transferred_bytes` 可能只改变报表，未改变实际仿真流量。优先用正确的张量和读比例表达访问模式，并用测试核对报表与计时。

例如权重只读取 1/8，但完整权重仍要常驻：

```python
from fractions import Fraction

kernel.weight_read_fraction = Fraction(1, 8)
```

这样降低 `loaded_weight_bytes`，不会缩小权重张量或其驻留存储。量化存储时还要计入 scale/zero-point，不能只将元素数乘以 4/8 bit。

混合精度计算可实现 `flops_by_dtype`：

```python
@property
def flops_by_dtype(self):
    return {"bf16": self.matmul_flops, "fp32": self.reduction_flops}
```

仿真器将各精度段按设备对应峰值串行累加。硬件没有某个非零算量 dtype 的吞吐配置时会报错。没有该属性时，计算 dtype 依次由 `w_dtype`、`dtype_`、第一个输入张量推断；新模型应明确精度，避免把存储精度误当计算精度。含独立串行子调用的复合算子应拆成多个核并连依赖，不应仅相加全部 FLOPs/流量后套一次 Roofline。

聚合访存模型可用 `Tensor(..., storage_bytes=0)` 描述累计传输，但此张量不再计入物理内存分配。此时必须另外建立真实工作集/临时内存的容量口径，并在结果中标注近似。MAGI `bridge.py` 的 `AnalyticalKernel` 使用这条路径；普通新算子优先采用真实形状张量。

## 4. 连接图、设备与通信

```python
from rooflang.language.graph import ComputeGraph
from rooflang.language.placement import Placement
from rooflang.programs.experiments.simulation import simulate

graph = ComputeGraph()
graph.add_kernel(kernel)
placement = Placement(hardware, graph)
placement.set_kernel_device(kernel, gpu)
graph.validate()
placement.validate(graph)
result = simulate(graph, placement, hardware)
```

其中 `hardware`、`gpu` 和 `kernel` 由模型构建函数提供；上面是该函数内的连接片段。`set_kernel_device()` 会为张量分配本地内存，远端存储或特殊放置用 `set_tensor_memory()` 明确指定。

后继算子消费前驱输出时，使用同一个输出张量，并加入 `graph.add_data_edge(producer, consumer, {"y": "x"})`。只有执行顺序约束时使用 `add_control_edge()`。控制边不会自动传输或延长数据张量寿命。

通信核从端口张量的内存放置推断参与设备；需要为所有输入/输出端口设置内存，并满足一对多、多对一或多对多的数量约束。核的 `total_bytes`、每 rank payload、算法传输量及 CSV 边界流量可能不是同一数值，必须说明，不能在公式和图中重复乘通信次数。

## 5. 验证后再纳入正式模型

| 测试 | 检查什么 |
| --- | --- |
| 手算小样本 | 固定尺寸得到独立计算的 FLOPs、读写字节、存储字节；至少有一个非默认尺寸 |
| 精度/非法输入 | 不同存储或计算 dtype、不支持的维度、非法分片明确处理，避免零值静默通过 |
| 图与硬件 | 放置完整、数据边一致、通信端口正确；可计算出预期时间的小硬件能得到相同结果 |
| 重复与共享 | 重复步骤放大算量/流量/耗时，共享权重与内存峰值不被调用次数误乘 |
| 导出 | CSV 行中的公式、dtype、`graph_multiplier` 与实际仿真一致；未标定工作没有伪造耗时 |

示例内的 `ExampleChecks` 包含这条流程的最小可运行版本；正式公共算子测试放在 `tests/unit_tests/language/`，模型专属测试放在 `tests/unit_tests/programs/`。不要把某次仿真的耗时常量硬编码成算子公式，也不要从参考输出读取数值来代替运算。

```bash
python examples/toy_model.py --check
python -m pytest -q tests/unit_tests/language tests/unit_tests/runtime
```

pytest 需要按 README 安装开发依赖。模型专属算子还应运行相应模型测试以及一组完整工作流输出检查。
