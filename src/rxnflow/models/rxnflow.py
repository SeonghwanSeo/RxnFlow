"""RxnFlow dynamic reaction policy and state-flow model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rxnflow.config import ModelConfig
from rxnflow.envs.chemistry.features import (
    FINGERPRINT_DIM,
    PROPERTY_DIM,
    PROPERTY_SCALE,
)
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.graph import BOND_FEATURE_DIM, NODE_FEATURE_DIM, GraphBatch

from .mpnn import MPNN
from .nn import mlp


class RxnFlowModel(nn.Module):
    def __init__(self, env: SynthesisEnv, config: ModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.env = env
        self.register_buffer(
            "property_scale", torch.tensor(PROPERTY_SCALE), persistent=False
        )
        self.graph_encoder = MPNN(
            node_dim=NODE_FEATURE_DIM,
            edge_dim=BOND_FEATURE_DIM,
            mol_feature_dim=PROPERTY_DIM,
            hidden_dim=hidden,
            num_layers=config.num_layers,
            max_reactions=env.max_reactions,
        )
        # Mean pooling and the virtual node can have different scales.
        self.mean_norm = nn.LayerNorm(hidden)
        self.virtual_norm = nn.LayerNorm(hidden)
        self.action_embedding = nn.Embedding(len(env.action_names), hidden)
        self.block_type_embedding = nn.Embedding(len(env.block_types), config.block_dim)
        # Project each feature, then normalize only in the fusion MLP.
        # Properties are scaled before projection in encode_block_features.
        self.fingerprint_encoder = nn.Linear(FINGERPRINT_DIM, config.block_dim)
        self.property_encoder = nn.Linear(PROPERTY_DIM, config.block_dim)
        self.block_encoder = mlp(
            config.block_dim * 3,
            config.block_dim,
            config.block_dim,
            config.block_mlp_layers,
            layernorm=True,
        )
        # Concatenate the 2H state and H reaction embeddings. Each head learns
        # their joint projection while the graph encoding is shared by reactions.
        self.first_block_head = mlp(
            3 * hidden,
            hidden,
            config.block_dim,
            config.mlp_layers,
            layernorm=True,
            dropout=config.dropout,
        )
        self.bi_reaction_head = mlp(
            3 * hidden,
            hidden,
            config.block_dim,
            config.mlp_layers,
            layernorm=True,
            dropout=config.dropout,
        )
        self.uni_reaction_head = mlp(
            3 * hidden,
            hidden,
            1,
            config.mlp_layers,
            layernorm=True,
            dropout=config.dropout,
        )
        # HSX main SimilarityMDP(dot): normalize only block embeddings and learn
        # a bounded temperature per reaction. Unary logits use the same scale
        # convention because all Uni/Bi choices share one categorical policy.
        self.min_temperature = 0.01
        self.max_temperature = 10.0
        self.logit_temperature = nn.Parameter(torch.empty(len(env.action_names)))
        self.log_z = nn.Parameter(torch.empty(()))
        self.init_weight()

    def init_weight(self) -> None:
        # Keep policy outputs nonzero so both dot-product branches receive
        # gradients from the first update. All Linear layers share this rule.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()
        # HSX main initializes both reaction and block-type embeddings small.
        nn.init.uniform_(self.action_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.block_type_embedding.weight, -0.1, 0.1)
        # HSX main ModelConfig initializes SimilarityMDP at 0.2.
        initial = (0.2 - self.min_temperature) / (
            self.max_temperature - self.min_temperature
        )
        nn.init.constant_(self.logit_temperature, math.log(initial / (1.0 - initial)))
        nn.init.zeros_(self.log_z)

    def encode_graphs(self, batch: GraphBatch) -> Tensor:
        mean, virtual = self.graph_encoder(batch).chunk(2, dim=-1)
        return torch.cat([self.mean_norm(mean), self.virtual_norm(virtual)], dim=-1)

    @property
    def temperature(self) -> Tensor:
        return (
            self.min_temperature
            + (self.max_temperature - self.min_temperature)
            * self.logit_temperature.sigmoid()
        )

    def score_scalar(self, state_embedding: Tensor, action_name: str) -> Tensor:
        return self.action_query(state_embedding, action_name)[0, 0]

    def _encode_blocks(
        self, block_type: str, indices: Tensor, device: torch.device
    ) -> Tensor:
        library = self.env.blocks[block_type]
        cpu_indices = indices.detach().cpu().to(torch.long)
        fingerprints = library.fingerprints[cpu_indices].to(device, dtype=torch.float32)
        properties = library.properties[cpu_indices].to(device)
        type_index = self.env.block_type_to_index[block_type]
        types = torch.full((len(indices),), type_index, dtype=torch.long, device=device)
        return self.encode_block_features(fingerprints, properties, types)

    def encode_block_features(
        self, fingerprints: Tensor, properties: Tensor, types: Tensor
    ) -> Tensor:
        properties = properties / self.property_scale
        return self.block_encoder(
            torch.cat(
                [
                    self.fingerprint_encoder(fingerprints),
                    self.property_encoder(properties),
                    self.block_type_embedding(types),
                ],
                dim=-1,
            )
        )

    def action_query(self, states: Tensor, action_name: str) -> Tensor:
        index = self.env.action_to_index[action_name]
        reaction = self.action_embedding.weight[index].expand(states.shape[0], -1)
        conditioned = torch.cat([states, reaction], dim=-1)
        if action_name == "first_block":
            head = self.first_block_head
        elif action_name in self.env.uni_reactions:
            head = self.uni_reaction_head
        else:
            head = self.bi_reaction_head
        return head(conditioned) / self.temperature[index]

    def score_blocks(
        self,
        state_embedding: Tensor,
        action_name: str,
        block_type: str,
        indices: Tensor,
    ) -> Tensor:
        assert indices.ndim == 1 and state_embedding.shape[0] == 1
        query = self.action_query(state_embedding, action_name)
        blocks = self._encode_blocks(block_type, indices, state_embedding.device)
        return F.normalize(blocks, dim=-1) @ query.squeeze(0)
