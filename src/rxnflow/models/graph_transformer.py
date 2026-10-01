"""Native PyTorch edge-aware transformer for fixed-size molecular graphs."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rxnflow.envs.graph import GraphBatch


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
        bias = self.edge_bias(edge_features).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~allowed.unsqueeze(1), torch.finfo(x.dtype).min)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
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
        max_reactions: int,
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
        self.reaction_count_embedding = nn.Embedding(max_reactions + 1, hidden_dim)
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, edge_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        # HSX readout: concatenate the molecular mean with the virtual node,
        # then project. A learned pooling gate would change that aggregation.
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim)
        )

    def forward(self, batch: GraphBatch) -> Tensor:
        nodes = self.node_projection(batch.node_features) * batch.node_mask.unsqueeze(-1)
        reaction_count = self.reaction_count_embedding(batch.reaction_count)
        mol_condition = self.mol_projection(batch.mol_features)
        capacity = self.capacity_projection(batch.remaining_capacity.unsqueeze(-1))
        condition = self.condition_norm(mol_condition + capacity + reaction_count)
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
        allowed[:, :node_count, node_count] = batch.node_mask
        allowed[:, node_count, :node_count] = batch.node_mask
        # GraphBatch has no padding edges. Every query gets one self edge,
        # including padded queries whose updates are zeroed by valid.
        allowed.diagonal(dim1=-2, dim2=-1).fill_(True)

        edges = torch.zeros(
            (batch_size, length, length, self.edge_dim), dtype=x.dtype, device=x.device
        )
        edges[:, :node_count, :node_count] = batch.bond_features
        for layer in self.layers:
            x = layer(x, valid, allowed, edges)
        x = self.output_norm(x)
        node_values = x[:, :node_count]
        node_count = batch.node_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        pooled = (node_values * batch.node_mask.unsqueeze(-1)).sum(dim=1) / node_count
        return self.readout(torch.cat([pooled, x[:, -1]], dim=-1))
