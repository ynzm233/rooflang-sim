"""All-Gather communication operator."""
from typing import List, Any
from .common_func import get_level_op_size_stage
from .core import OpType, Registry
from .base_comm import CommunicationOperator


@Registry.register_operator("all_gather", op_type=OpType.COMMUNICATION)
class AllGatherOperator(CommunicationOperator):
    """All-Gather communication operator."""

    def __init__(self, op_size, net_list, **kwargs: Any):
        super().__init__(op_size, net_list, **kwargs)
        self.op_name = "all_gather"

    def get_network_levels_op_size(self) -> List[float]:
        return get_level_op_size_stage(self.op_size, self.net_list)

    def aggregate(self, level_op_time: List[float]):
        return sum(level_op_time) if level_op_time else 0.0
