"""Base communication operator with network timing logic."""
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import math
from .core import OpType, Registry, Operator
from .core import get_dtype_size


class CommunicationOperator(Operator):
    """Base class for communication operators."""
    def __init__(self, op_size, net_list, **kwargs: Any) -> None:
        super().__init__(op_size=op_size, net_list=net_list, **kwargs)
        self.op_name : str
        self.op_size = op_size
        self.net_list = net_list
        self.comm_size = math.prod(net_list)

    def parse_args(self, **kwargs: Any) -> None:
        """Parse and validate communication arguments.
        This handles common parameters for all communication operators.
        Derived classes should call super().parse_args() first,
        then handle their specific parameters.
        """
        pass

    def get_flops(self) -> float:
        """Communication operators typically have minimal FLOPs."""
        return 0.0

    def get_memory_bytes(self) -> int:
        return self.op_size * 2  # Send and receive

    def get_memory_details(self) -> Dict[str, int]:
        return {}

    @abstractmethod
    def get_network_levels_op_size(self) -> List[float]:
        pass

    @abstractmethod
    def aggregate(self, level_op_time: List[float]) -> float:
        pass

    def op_quant_size(self) -> int:
        self.op_size *= get_dtype_size(self.communicate_dtype)