"""Cost recipe adapted from model_Sim; see SOURCES.json."""
from __future__ import annotations
import math
from typing import Any, Dict, List
from collections import Counter
from dataclasses import replace
from .compat import *
"""Embedding token block for LLM models."""

@BlockFactory.register('emb-token')
class TokenEmbeddingBlock(BaseBlock):
    """
    Token Embedding Block.

    Models an embedding lookup layer using a custom EmbeddingOperator with FLOPs=0.
    The weight matrix has dimensions vocab_size x hidden_size.
    When tie_word_embeddings=True, weight memory is not counted separately
    since it is tied with the lm_head projection layer.
    """

    def __init__(self, block_id: str, block_config: dict, comp_config: dict):
        super().__init__(block_id, block_config, comp_config)
        self.hidden_size = self.comp_config.hidden_size
        self.vocab_size = self.comp_config.vocab_size
        self.tie_word_embeddings = self.comp_config.tie_word_embeddings

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> List[Node]:
        self._pre_build(metadata.deploy_strategies)
        self.block_optag = OpTag.PREFILL.value
        batch_seq = metadata.batch_size * metadata.seq_len
        tp = metadata.parallel_strategies.get(Parallel.TP, 1)
        self.nodes.append(Node(node_name='TokenEmbedding', operator=EmbeddingOperator(batch_seq=batch_seq, vocab_size=self.vocab_size // tp, hidden_size=self.hidden_size, tie_word_embeddings=self.tie_word_embeddings), op_type=OpType.COMPUTE, module=ModuleType.EMBEDDING, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        if tp > 1:
            self.nodes.append(Node(node_name='Emb_TP_AllReduce', operator=AllReduceOperator(op_size=batch_seq * self.hidden_size, net_list=metadata.network_list[Parallel.TP]), op_type=OpType.COMMUNICATION, module=ModuleType.COMMUNICATION, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))
        return self.nodes
