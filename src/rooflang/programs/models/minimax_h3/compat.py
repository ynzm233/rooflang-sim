"""Small recipe vocabulary; graph execution belongs to RoofLang."""
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from rooflang.programs.models.magi2_preview.operators import (
    AddOperator, MulOperator, MatmulOperator, Conv1dOperator, Conv3dOperator,
    FlashAttentionOperator, RmsNormOperator, SiluOperator, RopeOperator,
    AllToAllOperator, AllReduceOperator, OpType, QuantConfig,
)
from rooflang.programs.models.magi2_preview._cost.core import Operator
from .cost_layernorm import LayerNormOperator
from .cost_gelu import GeluOperator
from .cost_embedding import EmbeddingOperator

class Parallel(Enum):
    SP = 'sp'
    CP = 'cp'
    TP = 'tp'
    DP = 'dp'

class ModuleType(Enum):
    OTHER = 'other'
    TEXTENCODER = 'textencoder'
    VAEDECODER = 'vaedecoder'
    ATTENTION = 'attention'
    MLP = 'mlp'
    COMMUNICATION = 'communication'
    MERGER = 'merger'
    EMBEDDING = 'embedding'

class OpTag(Enum):
    PREFILL = 1
    AF_COMM = 32

def merge_tags(tags):
    return sum(t.value for t in tags)

@dataclass
class Node:
    node_name: str
    operator: object
    op_type: OpType
    module: ModuleType
    tag_mask: int
    layer_idx: int
    block_id: str

class BaseBlock:
    def __init__(self, block_id, block_config, comp_config):
        self.block_id = comp_config['component_type'] + '.' + block_id
        self.block_config = SimpleNamespace(**{
            'type': '', 'num_heads': None, 'kv_heads': None, 'head_dim': None,
            'causal': True, 'use_adaLN': False, 'use_rope': True,
            'gate': False, 'hidden_act': None, 'intermediate_size': None,
            **block_config})
        self.comp_config = SimpleNamespace(**{'tie_word_embeddings': False, **comp_config})
        self.nodes = []
    def _pre_build(self, strategy):
        tags = {'llm': 3, 'vit': 128, 'dit': 256, 'vae_decoder': 1024, 'text_encoder': 2048}
        self.block_optag = next((v for k,v in tags.items() if self.block_id.startswith(k)), 0)

class BlockFactory:
    registry = {}
    @classmethod
    def register(cls, key):
        def wrap(block):
            cls.registry[key] = block
            return block
        return wrap

class M2NOperator(Operator):
    """Source no-op AF boundary when its deployment dimension is one."""
    def parse_args(self, **kwargs):
        self.op_size = 0
        self.net_list = [1]
    def get_flops(self): return 0.0
    def get_memory_details(self): return {}
