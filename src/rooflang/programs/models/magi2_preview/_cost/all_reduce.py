"""All-Reduce communication operator."""
from typing import List, Any
from .common_func import get_level_op_size_stage
from .core import OpType, Registry
from .base_comm import CommunicationOperator


@Registry.register_operator("all_reduce", op_type=OpType.COMMUNICATION)
class AllReduceOperator(CommunicationOperator):

    def __init__(self, op_size, net_list, **kwargs: Any):
        super().__init__(op_size, net_list, **kwargs)
        self.op_name = "all_reduce"

    def get_network_levels_op_size(self) -> List[float]:
        level_op_size = get_level_op_size_stage(self.op_size, self.net_list)
        return level_op_size

    def aggregate(self, level_op_time: List[float]):
        # 一次reduce_scatter + all_gather
        return sum(level_op_time) * 2 if level_op_time else 0.0

    def get_flops(self) -> float:
        """All-Reduce involves reduction operations."""
        return self.op_size * (self.comm_size - 1)