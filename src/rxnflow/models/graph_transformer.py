"""Native PyTorch edge-aware transformer for fixed-size molecular graphs."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from rxnflow.data.graph import GraphBatch


class EdgeAwareAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, edge_dim: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.edge_bias = nn.Linear(edge_dim, num_heads, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, valid: Tensor, allowed: Tensor, edge_features: Tensor
    ) -> Tensor:
        batch, length, hidden = x.shape
        normalized = self.norm(x)
        qkv = self.qkv(normalized).view(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = self.edge_bias(edge_features).permute(0, 3, 1, 2)
        scores = scores + bias
        scores = scores.masked_fill(~allowed.unsqueeze(1), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(self.dropout(attention), value)
        attended = attended.transpose(1, 2).reshape(batch, length, hidden)
        attended = self.output(attended) * valid.unsqueeze(-1)
        return x + self.dropout(attended)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, edge_dim: int, dropout: float):
        super().__init__()
        self.attention = EdgeAwareAttention(hidden_dim, num_heads, edge_dim, dropout)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, valid: Tensor, allowed: Tensor, edge_features: Tensor
    ) -> Tensor:
        x = self.attention(x, valid, allowed, edge_features)
        update = self.feed_forward(self.ff_norm(x)) * valid.unsqueeze(-1)
        return x + self.dropout(update)


class GraphTransformer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        mol_feature_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        num_workflows: int,
        max_protocols: int,
        num_action_kinds: int,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.node_projection = nn.Linear(node_dim, hidden_dim)
        self.mol_projection = nn.Sequential(
            nn.Linear(mol_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.capacity_projection = nn.Linear(1, hidden_dim)
        self.workflow_embedding = nn.Embedding(num_workflows + 1, hidden_dim)
        self.order_embedding = nn.Embedding(max_protocols + 2, hidden_dim)
        self.action_embedding = nn.Embedding(num_action_kinds, hidden_dim)
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, edge_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pool_gate = nn.Linear(hidden_dim, 1)

    def forward(self, batch: GraphBatch) -> Tensor:
        nodes = self.node_projection(batch.node_features) * batch.node_mask.unsqueeze(-1)
        workflow = self.workflow_embedding((batch.workflow_index + 1).clamp(min=0))
        order = self.order_embedding((batch.protocol_order + 1).clamp(min=0))
        action = self.action_embedding(batch.action_kind)
        mol_condition = self.mol_projection(batch.mol_features)
        capacity = self.capacity_projection(batch.remaining_capacity.unsqueeze(-1))
        condition = self.condition_norm(
            mol_condition + capacity + workflow + order + action
        )
        x = torch.cat([nodes, condition.unsqueeze(1)], dim=1)

        batch_size, node_count = batch.node_mask.shape
        valid = torch.cat(
            [
                batch.node_mask,
                torch.ones((batch_size, 1), dtype=torch.bool, device=x.device),
            ],
            dim=1,
        )
        length = node_count + 1
        allowed = torch.zeros(
            (batch_size, length, length), dtype=torch.bool, device=x.device
        )
        allowed[:, :node_count, :node_count] = batch.adjacency
        node_eye = torch.eye(node_count, dtype=torch.bool, device=x.device).unsqueeze(0)
        allowed[:, :node_count, :node_count] |= node_eye & batch.node_mask.unsqueeze(1)
        allowed[:, :node_count, node_count] = batch.node_mask
        allowed[:, node_count, :node_count] = batch.node_mask
        allowed[:, node_count, node_count] = True
        # Invalid padded queries attend to themselves, then are zeroed after every update.
        allowed |= torch.eye(length, dtype=torch.bool, device=x.device).unsqueeze(0)
        allowed &= valid.unsqueeze(1) | torch.eye(
            length, dtype=torch.bool, device=x.device
        ).unsqueeze(0)

        edges = torch.zeros(
            (batch_size, length, length, self.edge_dim), dtype=x.dtype, device=x.device
        )
        edges[:, :node_count, :node_count] = batch.bond_features
        for layer in self.layers:
            x = layer(x, valid, allowed, edges)
        x = self.output_norm(x)
        node_values = x[:, :node_count]
        gates = (
            self.pool_gate(node_values).squeeze(-1).masked_fill(~batch.node_mask, -1e9)
        )
        weights = torch.softmax(gates, dim=-1) * batch.node_mask
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = (node_values * (weights / denominator).unsqueeze(-1)).sum(dim=1)
        has_nodes = batch.node_mask.any(dim=1, keepdim=True)
        return x[:, -1] + torch.where(has_nodes, pooled, torch.zeros_like(pooled))
