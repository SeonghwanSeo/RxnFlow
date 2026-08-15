"""RxnFlow policy and state-flow model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from rxnflow.chemistry import (
    FINGERPRINT_DIM,
    PROPERTY_DIM,
    normalize_molecular_properties,
)
from rxnflow.config import ModelConfig
from rxnflow.data.graph import BOND_FEATURE_DIM, NODE_FEATURE_DIM, GraphBatch
from rxnflow.envs import SynthesisEnv

from .graph_transformer import GraphTransformer


class RxnFlowModel(nn.Module):
    def __init__(self, env: SynthesisEnv, config: ModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        max_protocols = max(len(workflow.protocols) for workflow in env.workflows)
        max_tier = max(int(library.tiers.max()) for library in env.blocks.values())
        self.env = env
        self.graph_encoder = GraphTransformer(
            node_dim=NODE_FEATURE_DIM,
            edge_dim=BOND_FEATURE_DIM,
            mol_feature_dim=PROPERTY_DIM,
            hidden_dim=hidden,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_workflows=len(env.workflows),
            max_protocols=max_protocols,
            num_action_kinds=4,
        )
        self.workflow_head = nn.Linear(hidden, len(env.workflows))
        self.protocol_embedding = nn.Embedding(len(env.protocol_names), hidden)
        self.block_type_embedding = nn.Embedding(len(env.block_types), hidden // 4)
        self.tier_embedding = nn.Embedding(max_tier + 1, hidden // 4)
        block_input = FINGERPRINT_DIM + PROPERTY_DIM + hidden // 2
        self.block_encoder = nn.Sequential(
            nn.Linear(block_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.block_query = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.log_z = nn.Parameter(torch.tensor(0.0))

    def encode_graphs(self, batch: GraphBatch) -> Tensor:
        return self.graph_encoder(batch)

    def score_workflows(self, state_embedding: Tensor) -> Tensor:
        return self.workflow_head(state_embedding)

    def _encode_blocks(
        self, block_type: str, indices: Tensor, device: torch.device
    ) -> Tensor:
        library = self.env.blocks[block_type]
        cpu_indices = indices.detach().cpu().to(torch.long)
        fingerprints = library.fingerprints[cpu_indices].to(device)
        properties = normalize_molecular_properties(library.properties[cpu_indices]).to(device)
        tiers = library.tiers[cpu_indices].to(device)
        type_index = self.env.block_type_to_index[block_type]
        types = torch.full_like(tiers, type_index)
        return self.block_encoder(
            torch.cat(
                [
                    fingerprints,
                    properties,
                    self.block_type_embedding(types),
                    self.tier_embedding(tiers),
                ],
                dim=-1,
            )
        )

    def score_blocks(
        self,
        state_embedding: Tensor,
        protocol_name: str,
        block_type: str,
        indices: Tensor,
    ) -> Tensor:
        assert indices.ndim == 1 and state_embedding.shape[0] == 1
        protocol_index = torch.tensor(
            [self.env.protocol_to_index[protocol_name]],
            dtype=torch.long,
            device=state_embedding.device,
        )
        protocol = self.protocol_embedding(protocol_index)
        query = self.block_query(torch.cat([state_embedding, protocol], dim=-1))
        blocks = self._encode_blocks(block_type, indices, state_embedding.device)
        return torch.matmul(blocks, query.squeeze(0)) / math.sqrt(query.shape[-1])

    def score_one_block(
        self,
        state_embedding: Tensor,
        protocol_name: str,
        block_type: str,
        block_index: int,
    ) -> Tensor:
        index = torch.tensor(
            [block_index], dtype=torch.long, device=state_embedding.device
        )
        return self.score_blocks(state_embedding, protocol_name, block_type, index)[0]
