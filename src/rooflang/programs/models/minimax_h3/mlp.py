"""Cost recipe adapted from model_Sim; see SOURCES.json."""
from __future__ import annotations
import math
from typing import Any, Dict, List
from collections import Counter
from dataclasses import replace
from .compat import *
@BlockFactory.register('mlp-dense')
class DenseBlock(BaseBlock):

    def __init__(self, block_id: str, block_config: Dict[str, Any], comp_config: Dict[str, Any]):
        super().__init__(block_id, block_config, comp_config)
        self.hidden_size = self.comp_config.hidden_size
        self.hidden_act = self.block_config.hidden_act if self.block_config.hidden_act is not None else 'silu'
        self.intermediate_size = self.block_config.intermediate_size
        self.up_size = self.intermediate_size * 2 if self.hidden_act == 'silu' else self.intermediate_size
        self.use_adaLN = self.block_config.use_adaLN

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> List[Node]:
        self.af_comm_tag = merge_tags([OpTag.AF_COMM])
        self._pre_build(metadata.deploy_strategies)
        batch_seq = metadata.batch_size * metadata.seq_len
        tp = metadata.parallel_strategies.get(Parallel.TP, 1)
        if self.intermediate_size % tp != 0:
            raise ValueError(f'intermediate_size ({self.intermediate_size}) must be divisible by TP degree ({tp})')
        self.nodes.append(Node(node_name='AF_split_comm_a2f', operator=M2NOperator(op_size=metadata.batch_size * metadata.seq_len * self.hidden_size, net_list=metadata.network_list['AF_COMM']), op_type=OpType.COMMUNICATION, module=ModuleType.COMMUNICATION, tag_mask=self.af_comm_tag, layer_idx=layer_idx, block_id=self.block_id))
        self.nodes.append(Node(node_name='Mlp_RmsNorm', operator=RmsNormOperator(batch_seq=batch_seq, dim=self.hidden_size), op_type=OpType.COMPUTE, module=ModuleType.MLP, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if self.use_adaLN:
            self.nodes.append(Node(node_name='Mlp_Add_Scale', operator=AddOperator(element_count_x=batch_seq * self.hidden_size, element_count_y=1), op_type=OpType.COMPUTE, module=ModuleType.MLP, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
            self.nodes.append(Node(node_name='Mlp_Mul_Scale', operator=MulOperator(element_count_x=batch_seq * self.hidden_size, element_count_y=batch_seq * self.hidden_size), op_type=OpType.COMPUTE, module=ModuleType.MLP, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
            self.nodes.append(Node(node_name='Mlp_Add_Shift', operator=AddOperator(element_count_x=batch_seq * self.hidden_size, element_count_y=batch_seq * self.hidden_size), op_type=OpType.COMPUTE, module=ModuleType.MLP, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        self.nodes.append(Node(node_name=f'Mlp_Up_{self.hidden_act}', operator=MatmulOperator(batch_seq=batch_seq, in_features=self.hidden_size, out_features=self.up_size // tp), op_type=OpType.COMPUTE, module=ModuleType.MLP, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if self.hidden_act == 'silu':
            self.nodes.append(Node(node_name=f'Mlp_Silu', operator=SiluOperator(batch_seq=batch_seq, dim=self.intermediate_size // tp), op_type=OpType.COMPUTE, module=ModuleType.MLP, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
            self.nodes.append(Node(node_name=f'Mlp_Mul', operator=MulOperator(element_count_x=batch_seq * self.intermediate_size // tp, element_count_y=batch_seq * self.intermediate_size // tp), op_type=OpType.COMPUTE, module=ModuleType.MLP, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        elif self.hidden_act == 'gelu':
            self.nodes.append(Node(node_name=f'Mlp_Gelu', operator=GeluOperator(batch_seq=batch_seq, dim=self.intermediate_size // tp), op_type=OpType.COMPUTE, module=ModuleType.MLP, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        else:
            raise Exception('Other activation functions need to be adapted !')
        self.nodes.append(Node(node_name=f'Mlp_Down_{self.hidden_act}', operator=MatmulOperator(batch_seq=batch_seq, in_features=self.intermediate_size // tp, out_features=self.hidden_size), op_type=OpType.COMPUTE, module=ModuleType.MLP, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if tp > 1:
            self.nodes.append(Node(node_name='Mlp_TP_AllReduce', operator=AllReduceOperator(op_size=batch_seq * self.hidden_size, net_list=metadata.network_list[Parallel.TP]), op_type=OpType.COMMUNICATION, module=ModuleType.COMMUNICATION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if self.use_adaLN:
            self.nodes.append(Node(node_name='Mlp_AdaLN_Gate', operator=MulOperator(element_count_x=batch_seq * self.hidden_size, element_count_y=batch_seq * self.hidden_size), op_type=OpType.COMPUTE, module=ModuleType.MLP, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        self.nodes.append(Node(node_name='Mlp_Residual_add', operator=AddOperator(element_count_x=batch_seq * self.hidden_size, element_count_y=batch_seq * self.hidden_size), op_type=OpType.COMPUTE, module=ModuleType.MLP, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        self.nodes.append(Node(node_name='AF_split_comm_f2a', operator=M2NOperator(op_size=metadata.batch_size * metadata.seq_len * self.hidden_size, net_list=metadata.network_list['AF_COMM']), op_type=OpType.COMMUNICATION, module=ModuleType.COMMUNICATION, tag_mask=self.af_comm_tag, layer_idx=layer_idx, block_id=self.block_id))
        return self.nodes
