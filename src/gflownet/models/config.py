from dataclasses import dataclass
from enum import Enum


@dataclass
class GraphTransformerConfig:
    num_heads: int = 2
    ln_type: str = "pre"
    num_mlp_layers: int = 0


class SeqPosEnc(int, Enum):
    Pos = 0
    Rotary = 1


@dataclass
class SeqTransformerConfig:
    num_heads: int = 2
    posenc: SeqPosEnc = SeqPosEnc.Rotary


@dataclass
class ModelConfig:
    """Generic configuration for models

    Attributes
    ----------
    num_layers : int
        The number of layers in the model
    num_emb : int
        The number of dimensions of the embedding
    """

    num_layers: int = 3
    num_emb: int = 128
    dropout: float = 0
    num_layers_building_block: int = 3
    num_emb_building_block: int = 128
    graph_transformer: GraphTransformerConfig = GraphTransformerConfig()
    graph_transformer_building_block: GraphTransformerConfig = GraphTransformerConfig()
    seq_transformer: SeqTransformerConfig = SeqTransformerConfig()
