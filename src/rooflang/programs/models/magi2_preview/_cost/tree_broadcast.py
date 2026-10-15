"""Blocking binary-tree broadcast along the slowest path of a network mesh."""
import math

from .core import QuantConfig
from .core import OpType, Registry
from .base_comm import CommunicationOperator


@Registry.register_operator("tree_broadcast", op_type=OpType.COMMUNICATION)
class TreeBroadcastOperator(CommunicationOperator):
    """Declared topology proxy; it does not predict NCCL algorithm selection."""

    def __init__(self, payload_bytes, net_list):
        if type(payload_bytes) is not int or payload_bytes < 0 or payload_bytes % 4:
            raise ValueError("broadcast payload must contain nonnegative whole 32-bit words")
        super().__init__(op_size=payload_bytes // 4, net_list=net_list,
                         quant_config=QuantConfig(compute_dtype="int32", weight_storage_dtype="int32",
                                                  kv_storage_dtype="int32", communicate_dtype="int32"))
        self.op_name = "p2p"
        self.shape = f"payload_bytes={payload_bytes}; ranks={math.prod(net_list)}"
        self.assumption = "blocking binary-tree critical path; actual NCCL algorithm unknown"

    def get_network_levels_op_size(self):
        return [self.op_size * math.ceil(math.log2(n)) if n > 1 else 0 for n in self.net_list]

    def aggregate(self, times):
        return sum(times)
