"""RxnFlow dynamic reaction policy and state-flow model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from rxnflow.config import ModelConfig
from rxnflow.envs import SynthesisEnv
from rxnflow.envs.chemistry.features import (
    FINGERPRINT_DIM,
    PROPERTY_DIM,
    PROPERTY_SCALE,
)
from rxnflow.envs.graph import BOND_FEATURE_DIM, NODE_FEATURE_DIM, GraphBatch

from .graph_transformer import GraphTransformer


class RxnFlowModel(nn.Module):
    def __init__(self, env: SynthesisEnv, config: ModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.env = env
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
        self.block_type_embedding = nn.Embedding(len(env.block_types), hidden // 2)
        block_input = FINGERPRINT_DIM + PROPERTY_DIM + hidden // 2
        self.block_encoder = nn.Sequential(
            nn.Linear(block_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.scalar_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.block_query = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # Distinct positional outcomes of the same reaction/block must have
        # distinct scores. Encode their product chemistry as well as the input
        # block; a reaction embedding alone cannot choose a regioisomer.
        self.outcome_encoder = nn.Sequential(
            nn.Linear(FINGERPRINT_DIM + PROPERTY_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.outcome_query = nn.Linear(hidden * 2, hidden)
        self.log_z = nn.Parameter(torch.tensor(0.0))

    def encode_graphs(self, batch: GraphBatch) -> Tensor:
        return self.graph_encoder(batch)

    def _action(self, name: str, device: torch.device) -> Tensor:
        index = torch.tensor(
            [self.env.action_to_index[name]], dtype=torch.long, device=device
        )
        return self.action_embedding(index)

    def score_scalar(self, state_embedding: Tensor, action_name: str) -> Tensor:
        action = self._action(action_name, state_embedding.device)
        return self.scalar_head(torch.cat([state_embedding, action], dim=-1))[0, 0]

    def _encode_blocks(
        self, block_type: str, indices: Tensor, device: torch.device
    ) -> Tensor:
        library = self.env.blocks[block_type]
        cpu_indices = indices.detach().cpu().to(torch.long)
        fingerprints = library.fingerprints[cpu_indices].to(device)
        properties = library.properties[cpu_indices].to(device)
        properties = properties / properties.new_tensor(PROPERTY_SCALE)
        type_index = self.env.block_type_to_index[block_type]
        types = torch.full((len(indices),), type_index, dtype=torch.long, device=device)
        return self.block_encoder(
            torch.cat(
                [fingerprints, properties, self.block_type_embedding(types)], dim=-1
            )
        )

    def score_blocks(
        self,
        state_embedding: Tensor,
        action_name: str,
        block_type: str,
        indices: Tensor,
    ) -> Tensor:
        assert indices.ndim == 1 and state_embedding.shape[0] == 1
        action = self._action(action_name, state_embedding.device)
        query = self.block_query(torch.cat([state_embedding, action], dim=-1))
        blocks = self._encode_blocks(block_type, indices, state_embedding.device)
        return torch.matmul(blocks, query.squeeze(0)) / math.sqrt(query.shape[-1])

    def score_outcomes(
        self,
        state_embedding: Tensor,
        action_name: str,
        properties: Tensor,
        fingerprints: Tensor,
    ) -> Tensor:
        device = state_embedding.device
        action = self._action(action_name, device)
        query = self.outcome_query(torch.cat([state_embedding, action], dim=-1))
        properties = properties.to(device)
        properties = properties / properties.new_tensor(PROPERTY_SCALE)
        outcomes = self.outcome_encoder(
            torch.cat(
                [
                    properties,
                    fingerprints.to(device),
                ],
                dim=-1,
            )
        )
        return outcomes @ query.squeeze(0) / math.sqrt(query.shape[-1])
