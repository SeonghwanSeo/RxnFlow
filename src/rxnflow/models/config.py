from dataclasses import dataclass, field

from gflownet.utils.misc import StrictDataClass


@dataclass
class GraphTransformerConfig(StrictDataClass):
    num_heads: int = 2
    ln_type: str = "pre"
    num_layers: int = 4  # NOTE: original: num_mlp_layers: int = 0, I think this is bug...
    concat_heads: bool = True


@dataclass
class ModelConfig(StrictDataClass):
    """Generic configuration for models

    Attributes
    ----------
    num_layers : int
        The number of layers in the model
    num_emb : int
        The number of dimensions of the embedding
    num_layers_building_block : int
        The number of layers in the action embedding
    num_emb_building_block : int
        The number of dimensions of the action embedding
    """

    num_mlp_layers: int = 2
    num_emb: int = 128
    dropout: float = 0
    num_mlp_layers_block: int = 1
    num_emb_block: int = 128
    graph_transformer: GraphTransformerConfig = field(default_factory=GraphTransformerConfig)
