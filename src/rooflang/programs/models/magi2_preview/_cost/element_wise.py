from dataclasses import dataclass, field
from typing import Any, List

from .core import Operator, OperatorParams, OpType, Registry
from .core import DTYPE_INFO, get_dtype_size

_ELEMENTWISE_FLOPS_PER_ELEMENT = 1

@dataclass
class ElementWiseParams(OperatorParams):
    element_count_x: int = field(default=0, metadata={"min": 0, "desc": "X向量中元素总个数"})
    element_count_y: int = field(default=0, metadata={"min": 0, "desc": "Y向量中元素总个数"})
    fused: List[bool] = field(default_factory=lambda: [False, False], metadata={"desc": "是否被融合"})
    input_dtype_x: str | None = None
    input_dtype_y: str | None = None
    output_dtype: str | None = None

class ElementWiseOperator(Operator):

    COMPUTE_UNIT = "vector"

    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.element_count_x = params.element_count_x if params is not None else kwargs.get("element_count_x", 0)
        self.element_count_y = params.element_count_y if params is not None else kwargs.get("element_count_y", 0)
        self.fused = params.fused if params is not None else kwargs.get("fused", [False, False])
        for name in ("input_dtype_x", "input_dtype_y", "output_dtype"):
            value = getattr(params, name, None) if params is not None else kwargs.get(name)
            if value is not None and value not in DTYPE_INFO:
                raise ValueError(f"unsupported elementwise boundary dtype: {value}")
            setattr(self, name, value)

    def get_flops(self) -> float:
        return float(max(self.element_count_x, self.element_count_y) * _ELEMENTWISE_FLOPS_PER_ELEMENT)

    def get_memory_details(self) -> dict[str, int]:
        a_sz = self.quant_config.get_compute_size()
        x_size = get_dtype_size(self.input_dtype_x) if self.input_dtype_x else a_sz
        y_size = get_dtype_size(self.input_dtype_y) if self.input_dtype_y else a_sz
        out_size = get_dtype_size(self.output_dtype) if self.output_dtype else a_sz
        input_bytes = 0 if self.fused[0] else self.element_count_x * x_size + self.element_count_y * y_size
        output_bytes = 0 if self.fused[1] else max(self.element_count_x, self.element_count_y) * out_size
        activation_bytes = input_bytes + output_bytes

        return self.memory_details(
            activation=activation_bytes
        )

@dataclass
class MulParams(ElementWiseParams):
    pass

@Registry.register_operator("mul", op_type=OpType.COMPUTE)
class MulOperator(ElementWiseOperator):

    PARAM_CLASS = MulParams

@dataclass
class SubParams(ElementWiseParams):
    pass

@Registry.register_operator("sub", op_type=OpType.COMPUTE)
class SubOperator(ElementWiseOperator):

    PARAM_CLASS = SubParams

@dataclass
class DivParams(ElementWiseParams):
    pass

@Registry.register_operator("div", op_type=OpType.COMPUTE)
class DivOperator(ElementWiseOperator):

    PARAM_CLASS = DivParams

@dataclass
class AddParams(ElementWiseParams):
    pass

@Registry.register_operator("add", op_type=OpType.COMPUTE)
class AddOperator(ElementWiseOperator):

    PARAM_CLASS = AddParams
