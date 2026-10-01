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

from .graph_transformer import GraphTransformer


class RxnFlowModel(nn.Module):
    def __init__(self, env: SynthesisEnv, config: ModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.env = env
        self.register_buffer(
            "property_scale", torch.tensor(PROPERTY_SCALE), persistent=False
        )
        self.graph_encoder = GraphTransformer(
            node_dim=NODE_FEATURE_DIM,
            edge_dim=BOND_FEATURE_DIM,
            mol_feature_dim=PROPERTY_DIM,
            hidden_dim=hidden,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            max_reactions=env.max_reactions,
        )
        self.action_embedding = nn.Embedding(len(env.action_names), hidden)
        self.block_type_embedding = nn.Embedding(len(env.block_types), hidden)
        # explore_250509: keep fingerprint and physical-property projections
        # separate before fusing them with the categorical block type. Price
        # tiers are deliberately absent from the public Enamine environment.
        self.fingerprint_encoder = nn.Sequential(
            nn.Linear(FINGERPRINT_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.property_encoder = nn.Sequential(
            nn.Linear(PROPERTY_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.block_encoder = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # Reaction conditioning follows the old HSX additive protocol embedding.
        # The graph is encoded once; competing reactions use these cheap heads.
        self.first_block_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.bi_reaction_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.uni_reaction_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # HSX main SimilarityMDP(dot): normalize only block embeddings and learn
        # a bounded temperature per reaction. Unary logits use the same scale
        # convention because all Uni/Bi choices share one categorical policy.
        self.min_temperature = 0.01
        self.max_temperature = 10.0
        initial = (1.0 - self.min_temperature) / (
            self.max_temperature - self.min_temperature
        )
        self.logit_temperature = nn.Parameter(
            torch.full((len(env.action_names),), math.log(initial / (1.0 - initial)))
        )
        nn.init.uniform_(self.action_embedding.weight, -1.0, 1.0)
        nn.init.uniform_(self.block_type_embedding.weight, -1.0, 1.0)
        self.log_z = nn.Parameter(torch.tensor(0.0))

    def encode_graphs(self, batch: GraphBatch) -> Tensor:
        return self.graph_encoder(batch)

    @property
    def temperature(self) -> Tensor:
        return (
            self.min_temperature
            + (self.max_temperature - self.min_temperature)
            * self.logit_temperature.sigmoid()
        )

    def score_scalar(self, state_embedding: Tensor, action_name: str) -> Tensor:
        index = self.env.action_to_index[action_name]
        conditioned = F.silu(state_embedding + self.action_embedding.weight[index])
        return self.uni_reaction_head(conditioned)[0, 0] / self.temperature[index]

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

    def action_queries(
        self, states: Tensor, action_indices: Tensor, positions: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """One head call per action kind, across all states and reactions.

        positions follows ActionKind order: FirstBlock, UniReaction, BiReaction.
        Temperatures divide queries before their dot product with block vectors.
        """
        conditioned = F.silu(states + self.action_embedding(action_indices))
        scale = self.temperature[action_indices]
        queries = torch.zeros_like(states)
        unary = states.new_zeros(len(states))
        first, uni, bi = positions
        for head, indices in (
            (self.first_block_head, first),
            (self.bi_reaction_head, bi),
        ):
            if len(indices):
                queries = queries.index_copy(
                    0, indices, head(conditioned[indices]) / scale[indices, None]
                )
        if len(uni):
            unary = unary.index_copy(
                0, uni, self.uni_reaction_head(conditioned[uni]).squeeze(-1) / scale[uni]
            )
        return queries, unary

    def score_blocks(
        self,
        state_embedding: Tensor,
        action_name: str,
        block_type: str,
        indices: Tensor,
    ) -> Tensor:
        assert indices.ndim == 1 and state_embedding.shape[0] == 1
        index = self.env.action_to_index[action_name]
        conditioned = F.silu(state_embedding + self.action_embedding.weight[index])
        head = (
            self.first_block_head
            if action_name == "first_block"
            else self.bi_reaction_head
        )
        query = head(conditioned)
        blocks = self._encode_blocks(block_type, indices, state_embedding.device)
        return F.normalize(blocks, dim=-1) @ query.squeeze(0) / self.temperature[index]
