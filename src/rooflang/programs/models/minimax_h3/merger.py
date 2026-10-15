"""Cost recipe adapted from model_Sim; see SOURCES.json."""
from __future__ import annotations
import math
from typing import Any, Dict, List
from collections import Counter
from dataclasses import replace
from .compat import *
@BlockFactory.register('patch-merger')
class PatchMergerBlock(BaseBlock):

    def __init__(self, block_id: str, block_config: Dict[str, Any], comp_config: Dict[str, Any]):
        super().__init__(block_id, block_config, comp_config)
        self.hidden_size = self.comp_config.hidden_size
        self.out_hidden_size = self.comp_config.out_hidden_size
        self.spatial_merge_size = self.comp_config.spatial_merge_size

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> List[Node]:
        self._pre_build(metadata.deploy_strategies)
        batch_seq = metadata.batch_size * metadata.seq_len
        self.nodes.append(Node(node_name='PatchMerger_LayerNorm', operator=LayerNormOperator(batch_seq=batch_seq, dim=self.hidden_size), op_type=OpType.COMPUTE, module=ModuleType.MERGER, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        self.nodes.append(Node(node_name='PatchMerger_Fc1_Linear', operator=MatmulOperator(batch_seq=batch_seq // self.spatial_merge_size ** 2, in_features=self.hidden_size * self.spatial_merge_size ** 2, out_features=self.hidden_size * self.spatial_merge_size ** 2), op_type=OpType.COMPUTE, module=ModuleType.MERGER, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        self.nodes.append(Node(node_name=f'PatchMerger_Gelu', operator=GeluOperator(batch_seq=batch_seq // self.spatial_merge_size ** 2, dim=self.hidden_size * self.spatial_merge_size ** 2), op_type=OpType.COMPUTE, module=ModuleType.MERGER, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        self.nodes.append(Node(node_name='PatchMerger_Fc2_Linear', operator=MatmulOperator(batch_seq=batch_seq // self.spatial_merge_size ** 2, in_features=self.hidden_size * self.spatial_merge_size ** 2, out_features=self.out_hidden_size), op_type=OpType.COMPUTE, module=ModuleType.MERGER, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        return self.nodes
