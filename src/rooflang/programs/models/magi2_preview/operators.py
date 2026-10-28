"""Cost-only MAGI-2 operators adapted from Markov_Infer_Sim.

The local formulas describe FLOPs, traffic, precision and ordered subcalls.
They do not import Markov_Infer_Sim or calculate execution time; the RoofLang
adapter lowers these descriptions to native execution-graph nodes.
"""

from ._cost.core import QuantConfig, get_dtype_size, OpType, Operator, ComputeComponent, CompositeOperator, OperatorStage
from ._cost.element_wise import AddOperator, SubOperator, MulOperator
from ._cost.matmul import MatmulOperator
from ._cost.groupmatmul import GroupMatmulOperator
from ._cost.rmsnorm import RmsNormOperator
from ._cost.rope import RopeOperator
from ._cost.silu import SiluOperator
from ._cost.reduce_sum import ReduceSumOperator
from ._cost.all_to_all import AllToAllOperator
from ._cost.all_gather import AllGatherOperator
from ._cost.all_reduce import AllReduceOperator
from ._cost.flash_attention_sink import FlashAttentionSinkOperator
from ._cost.flash_attention import FlashAttentionOperator
from ._cost.npu_fused_infer_attention_score import (
    NpuFusedInferAttentionScoreOperator, NpuFusedInferAttentionScoreParams,
)
from ._cost.flash_mh_moe import FlashMHMoEOperator
from ._cost.moe_routing import MoERoutingOperator
from ._cost.swiglu7 import SwiGLU7Operator
from ._cost.mhc import MHCOperator
from ._cost.tensor_move import TensorMoveOperator
from ._cost.interpolate import InterpolateOperator
from ._cost.feature_transform import FourierFeaturesOperator, SigmoidGateOperator, ReluOperator, WeightNormOperator
from ._cost.tree_broadcast import TreeBroadcastOperator
from ._cost.conv1d import Conv1dOperator
from ._cost.conv3d import Conv3dOperator
from ._cost.conv_transpose1d import ConvTranspose1dOperator
from ._cost.snake_beta import SnakeBetaOperator


__all__ = ['QuantConfig', 'get_dtype_size', 'OpType', 'Operator', 'ComputeComponent', 'CompositeOperator', 'OperatorStage', 'AddOperator', 'SubOperator', 'MulOperator', 'MatmulOperator', 'GroupMatmulOperator', 'RmsNormOperator', 'RopeOperator', 'SiluOperator', 'ReduceSumOperator', 'AllToAllOperator', 'AllGatherOperator', 'AllReduceOperator', 'FlashAttentionSinkOperator', 'FlashAttentionOperator', 'FlashMHMoEOperator', 'MoERoutingOperator', 'SwiGLU7Operator', 'MHCOperator', 'TensorMoveOperator', 'InterpolateOperator', 'FourierFeaturesOperator', 'SigmoidGateOperator', 'ReluOperator', 'WeightNormOperator', 'TreeBroadcastOperator', 'Conv1dOperator', 'Conv3dOperator', 'ConvTranspose1dOperator', 'SnakeBetaOperator']
__all__ += ['NpuFusedInferAttentionScoreOperator', 'NpuFusedInferAttentionScoreParams']
