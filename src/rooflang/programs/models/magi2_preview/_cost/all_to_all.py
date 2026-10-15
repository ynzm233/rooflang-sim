"""All-to-All communication operator."""
from typing import Any, List
from .common_func import get_level_op_size_mesh
from .core import OpType, Registry
from .base_comm import CommunicationOperator


@Registry.register_operator("all_to_all", op_type=OpType.COMMUNICATION)
class AllToAllOperator(CommunicationOperator):
    """All-to-All communication operator."""

    def __init__(self, op_size, net_list, **kwargs: Any):
        super().__init__(op_size, net_list, **kwargs)
        self.op_name = "all2all"

    def get_network_levels_op_size(self) -> List[float]:
        return get_level_op_size_mesh(self.op_size/self.comm_size, self.net_list)

    def aggregate(self, level_op_time: List[float]):
        return max(level_op_time) if level_op_time else 0.0