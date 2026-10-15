"""Embedding lookup operator.

Embedding is a memory-bound operation with no compute FLOPs.
The weight matrix has dimensions vocab_size x hidden_size.
It may share weights with the lm_head projection layer (tie_word_embeddings).
"""

from dataclasses import dataclass, field
from typing import Any

from rooflang.programs.models.magi2_preview._cost.core import Operator, OperatorParams, OpType, Registry


@dataclass
class EmbeddingParams(OperatorParams):
    """Embedding lookup parameters."""
    batch_seq: int = field(default=0, metadata={"min": 1, "desc": "Batch size * sequence length"})
    vocab_size: int = field(default=0, metadata={"min": 1, "desc": "Vocabulary size"})
    hidden_size: int = field(default=0, metadata={"min": 1, "desc": "Embedding dimension"})
    tie_word_embeddings: bool = field(default=False, metadata={"desc": "Whether to share weights with lm_head"})


@Registry.register_operator("embedding", op_type=OpType.COMPUTE)
class EmbeddingOperator(Operator):
    """Embedding lookup operator.

    Embedding is purely memory-bound. FLOPs = 0.
    When tie_word_embeddings=True, the weight memory is not counted separately
    (as it is already accounted for by the lm_head block).
    """

    COMPUTE_UNIT = "vector"
    PARAM_CLASS = EmbeddingParams

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def parse_args(self, **kwargs: Any) -> None:
        params = self.params
        self.batch_seq = params.batch_seq if params is not None else kwargs.get("batch_seq", 0)
        self.vocab_size = params.vocab_size if params is not None else kwargs.get("vocab_size", 0)
        self.hidden_size = params.hidden_size if params is not None else kwargs.get("hidden_size", 0)
        self.tie_word_embeddings = params.tie_word_embeddings if params is not None else kwargs.get("tie_word_embeddings", False)

    def get_flops(self) -> float:
        # Embedding lookup involves no arithmetic computation
        return 0.0

    def get_memory_details(self) -> dict[str, int]:
        w_sz = self.quant_config.get_weight_size()
        weight_saved_bytes = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size * w_sz
        weight_read = self.batch_seq * self.hidden_size
        return self.memory_details(
            weight=weight_read,
            weight_saved=weight_saved_bytes
        )
