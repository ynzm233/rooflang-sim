"""Cost recipe adapted from model_Sim; see SOURCES.json."""
from __future__ import annotations
import math
from typing import Any, Dict, List
from collections import Counter
from dataclasses import replace
from .compat import *
@BlockFactory.register('attn-mha')
class MHABlock(BaseBlock):
    """Multi-Head Attention (MHA): num_heads == kv_heads."""

    def __init__(self, block_id: str, block_config: Dict[str, Any], comp_config: Dict[str, Any]):
        super().__init__(block_id, block_config, comp_config)
        self.type = self.block_config.type
        self.hidden_size = self.comp_config.hidden_size
        self.num_heads = self.block_config.num_heads
        self.kv_heads = self.num_heads
        self.head_dim = self.block_config.head_dim
        self.causal = self.block_config.causal
        self.use_adaLN = self.block_config.use_adaLN
        self.use_rope = self.block_config.use_rope
        self.gate = self.block_config.gate

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> List[Node]:
        self._pre_build(metadata.deploy_strategies)
        batch_seq = metadata.batch_size * metadata.seq_len
        tp = metadata.parallel_strategies.get(Parallel.TP, 1)
        sp = metadata.parallel_strategies.get(Parallel.SP, 1)
        cp = metadata.parallel_strategies.get(Parallel.CP, 1)
        if self.num_heads % self.kv_heads != 0:
            raise ValueError(f'num_heads ({self.num_heads}) must be divisible by kv_heads ({self.kv_heads})')
        if self.kv_heads % (tp * sp) != 0:
            raise ValueError(f'kv_heads ({self.kv_heads}) must be divisible by tp * sp ({tp * sp})')
        self.nodes.append(Node(node_name='Attn_RmsNorm', operator=RmsNormOperator(batch_seq=batch_seq, dim=self.hidden_size), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if self.use_adaLN:
            self.nodes.append(Node(node_name='Attn_Add_Scale', operator=AddOperator(element_count_x=batch_seq * self.hidden_size, element_count_y=1), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
            self.nodes.append(Node(node_name='Attn_Mul_Scale', operator=MulOperator(element_count_x=batch_seq * self.hidden_size, element_count_y=batch_seq * self.hidden_size), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
            self.nodes.append(Node(node_name='Attn_Add_Shift', operator=AddOperator(element_count_x=batch_seq * self.hidden_size, element_count_y=batch_seq * self.hidden_size), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if self.gate:
            self.nodes.append(Node(node_name='Attn_QKVGLinear', operator=MatmulOperator(batch_seq=batch_seq, in_features=self.hidden_size, out_features=(self.num_heads * 2 + self.kv_heads * 2) * self.head_dim // tp), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        else:
            self.nodes.append(Node(node_name='Attn_QKVLinear', operator=MatmulOperator(batch_seq=batch_seq, in_features=self.hidden_size, out_features=(self.num_heads + self.kv_heads * 2) * self.head_dim // tp), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        self.nodes.append(Node(node_name='Attn_QNorm', operator=RmsNormOperator(batch_seq=batch_seq * self.num_heads // tp, dim=self.head_dim), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        self.nodes.append(Node(node_name='Attn_KNorm', operator=RmsNormOperator(batch_seq=batch_seq * self.kv_heads // tp, dim=self.head_dim), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if self.use_rope:
            self.nodes.append(Node(node_name='Attn_QRoPE', operator=RopeOperator(batch_seq=batch_seq, dim=self.num_heads * self.head_dim // tp), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
            self.nodes.append(Node(node_name='Attn_KRoPE', operator=RopeOperator(batch_seq=batch_seq, dim=self.kv_heads * self.head_dim // tp), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if sp > 1:
            self.nodes.append(Node(node_name='Attn_Q_AllToAll', operator=AllToAllOperator(op_size=batch_seq * self.num_heads * self.head_dim // tp, net_list=metadata.network_list[Parallel.SP]), op_type=OpType.COMMUNICATION, module=ModuleType.COMMUNICATION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
            self.nodes.append(Node(node_name='Attn_K_AllToAll', operator=AllToAllOperator(op_size=batch_seq * self.kv_heads * self.head_dim // tp, net_list=metadata.network_list[Parallel.SP]), op_type=OpType.COMMUNICATION, module=ModuleType.COMMUNICATION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
            self.nodes.append(Node(node_name='Attn_V_AllToAll', operator=AllToAllOperator(op_size=batch_seq * self.kv_heads * self.head_dim // tp, net_list=metadata.network_list[Parallel.SP]), op_type=OpType.COMMUNICATION, module=ModuleType.COMMUNICATION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if cp > 1:
            self.nodes.append(Node(node_name='Attn_P2P_CP', operator=PointToPointOperator(op_size=2 * batch_seq * self.kv_heads * self.head_dim // tp, net_list=metadata.network_list[Parallel.CP]), op_type=OpType.COMMUNICATION, module=ModuleType.COMMUNICATION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        oc = metadata.offload_config
        if metadata.cache_len > 0 and oc is not None and oc.prefix_cache.enabled:
            pc_total = metadata.batch_size * (self.kv_heads // tp) * metadata.cache_len * self.head_dim * 2
            self.nodes.append(Node(node_name='kv_load_prefix', operator=KVLoadOperator(total_bytes=pc_total, l2_hit_ratio=oc.prefix_cache.l2_hit_ratio, l3_hit_ratio=oc.prefix_cache.l3_hit_ratio), op_type=OpType.COMPUTE, module=ModuleType.KV_CACHE, tag_mask=OpTag.PREFILL.value, layer_idx=layer_idx, block_id=self.block_id))
        for idx in range(cp):
            global_kv_len = (metadata.kv_len - metadata.cache_len) * sp + metadata.cache_len
            self.nodes.append(Node(node_name=f'Attn_FA_{idx}', operator=FlashAttentionOperator(batch_size=metadata.batch_size, num_heads=self.num_heads // tp // sp, kv_heads=self.kv_heads // tp // sp, q_len=metadata.seq_len * sp, kv_len=global_kv_len, head_dim=self.head_dim, causal=self.causal), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if sp > 1:
            self.nodes.append(Node(node_name='Attn_O_AllToAll', operator=AllToAllOperator(op_size=batch_seq * self.num_heads * self.head_dim // tp, net_list=metadata.network_list[Parallel.SP]), op_type=OpType.COMMUNICATION, module=ModuleType.COMMUNICATION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if self.gate:
            self.nodes.append(Node(node_name='Attn_gate_sigmoid', operator=SigmoidOperator(batch_seq=batch_seq, dim=self.num_heads * self.head_dim // tp), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
            self.nodes.append(Node(node_name='Attn_gate_output_mul', operator=MulOperator(element_count_x=batch_seq * (self.num_heads * self.head_dim // tp), element_count_y=batch_seq * (self.num_heads * self.head_dim // tp)), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        self.nodes.append(Node(node_name='Attn_OLinear', operator=MatmulOperator(batch_seq=batch_seq, in_features=self.num_heads * self.head_dim // tp, out_features=self.hidden_size), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if tp > 1:
            self.nodes.append(Node(node_name='Attn_TP_AllReduce', operator=AllReduceOperator(op_size=batch_seq * self.hidden_size, net_list=metadata.network_list[Parallel.TP]), op_type=OpType.COMMUNICATION, module=ModuleType.COMMUNICATION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if self.use_adaLN:
            self.nodes.append(Node(node_name='Attn_AdaLN_Gate', operator=MulOperator(element_count_x=batch_seq * self.hidden_size, element_count_y=batch_seq * self.hidden_size), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        self.nodes.append(Node(node_name='Attn_Residual_add', operator=AddOperator(element_count_x=batch_seq * self.hidden_size, element_count_y=batch_seq * self.hidden_size), op_type=OpType.COMPUTE, module=ModuleType.ATTENTION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        return self.nodes

@BlockFactory.register('attn-mqa')
class MQABlock(MHABlock):
    """Multi-Query Attention (MQA): kv_heads == 1."""

    def __init__(self, block_id: str, block_config: Dict[str, Any], comp_config: Dict[str, Any]):
        super().__init__(block_id, block_config, comp_config)
        self.kv_heads = 1

@BlockFactory.register('attn-gqa')
class GQABlock(MHABlock):
    """Grouped Query Attention (GQA): kv_heads < num_heads."""

    def __init__(self, block_id: str, block_config: Dict[str, Any], comp_config: Dict[str, Any]):
        super().__init__(block_id, block_config, comp_config)
        self.kv_heads = self.block_config.kv_heads
        assert self.num_heads % self.kv_heads == 0
