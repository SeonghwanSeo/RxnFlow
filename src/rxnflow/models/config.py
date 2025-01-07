from dataclasses import dataclass, field

from gflownet.utils.misc import StrictDataClass


@dataclass
class GraphTransformerConfig(StrictDataClass):
    num_heads: int = 2
    ln_type: str = "pre"
    num_layers: int = 3
    concat_heads: bool = True


@dataclass
class ModelConfig(StrictDataClass):
    """Generic configuration for models

    Attributes
    ----------
    num_emb : int
        The number of dimensions of the embedding
    num_layers : int
        The number of layers in the model
    num_emb_block : int
        The number of dimensions of the action embedding
    num_mlp_layers_block : int
        The number of layers in the action embedding
    """

    num_emb: int = 128
    num_mlp_layers: int = 2
    num_emb_block: int = 64
    num_mlp_layers_block: int = 1
    graph_transformer: GraphTransformerConfig = field(default_factory=GraphTransformerConfig)
