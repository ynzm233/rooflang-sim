"""Group Matmul Projection operator. (For MOE Packed Experts)"""

from dataclasses import dataclass, field
from typing import Any, List

from .core import Operator, OperatorParams, OpType, Registry


@dataclass
class GroupMatmulParams(OperatorParams):
    """Group Matmul for PACKED multiple independent matmuls (MOE experts)"""
    """
    用途：将 N 个独立的专家 Matmul 打包成 1 个 GroupMatmul（MOE稀疏激活场景）
    - expert_token_list: 列表，长度=单卡总专家数
      元素=0 → 专家未激活
      元素>0 → 专家激活，值为该专家处理的token数
    - in_features: 单个专家的输入维度（所有专家相同）
    - out_features: 单个专家的输出维度（所有专家相同）
    """
    expert_token_list: List[int] = field(default_factory=list, metadata={"desc": "List of tokens per expert, 0 means inactive"})
    in_features: int = field(default=0, metadata={"min": 1, "desc": "Per-expert input dim"})
    out_features: int = field(default=0, metadata={"min": 1, "desc": "Per-expert output dim"})


@Registry.register_operator("group_matmul", op_type=OpType.COMPUTE)
class GroupMatmulOperator(Operator):
    """Packed Group Matmul for MOE Experts: N independent matmuls in one op (sparse activation)"""

    COMPUTE_UNIT = "cube"
    PARAM_CLASS = GroupMatmulParams

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        # 解析参数
        self.expert_token_list = params.expert_token_list if params is not None else kwargs.get("expert_token_list", [])
        self.in_features = params.in_features if params is not None else kwargs.get("in_features", 0)
        self.out_features = params.out_features if params is not None else kwargs.get("out_features", 0)

        # 自动计算：总专家数 / 激活专家数 / 所有激活专家的总token数
        self.total_experts = len(self.expert_token_list)
        self.active_expert_list = [t for t in self.expert_token_list if t > 0]
        self.active_experts = len(self.active_expert_list)
        self.total_active_tokens = sum(self.active_expert_list)

    @property
    def _num_output_elements(self) -> int:
        # 输出元素 = 所有激活专家的token总数 × 输出维度
        return self.total_active_tokens * self.out_features

    @property
    def _num_input_elements(self) -> int:
        # 输入元素 = 所有激活专家的token总数 × 输入维度
        return self.total_active_tokens * self.in_features

    @property
    def _num_weight_elements(self) -> int:
        # 权重 = 单卡全部专家的权重（无论是否激活）
        return self.total_experts * self.in_features * self.out_features

    def get_flops(self) -> float:
        # FLOPs = 2 * 输入维度 * 所有激活专家总token数 * 输出维度
        return float(2 * self.total_active_tokens * self.in_features * self.out_features)

    def get_memory_details(self) -> dict[str, int]:
        a_sz = self.quant_config.get_compute_size()
        w_sz = self.quant_config.get_weight_size()

        # 权重存储：全部专家
        weight_saved_bytes = self._num_weight_elements * w_sz

        # 权重读取：只读取激活专家
        weight_read_bytes = self.active_experts * self.in_features * self.out_features * w_sz

        # 激活态内存
        input_bytes = self._num_input_elements * a_sz
        output_bytes = self._num_output_elements * a_sz
        activation_bytes = input_bytes + output_bytes

        return self.memory_details(
            weight=weight_read_bytes,
            weight_saved=weight_saved_bytes,
            activation=activation_bytes
        )