from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rxnflow.envs.features import PROPERTY_SCALE
from rxnflow.envs.graph import GraphBatch

from .nn import mlp


class ConvLayer(nn.Module):
    """Modified GINEConv with pre-normalization, residual connection, and virtual node."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.norm_h = nn.LayerNorm(model_dim)
        self.norm_v = nn.LayerNorm(model_dim)
        self.v_to_h = nn.Linear(model_dim, model_dim)
        self.h_to_v = nn.Linear(model_dim, model_dim)
        self.mlp = mlp(model_dim, 2 * model_dim, model_dim, 2)

    def forward(
        self,
        h: torch.Tensor,
        v: torch.Tensor,
        e: torch.Tensor,
        neighbor_index: torch.Tensor,
        node_mask: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update atoms [B, N, D] and virtual nodes [B, D] simultaneously."""
        # Residual connection
        _h, _v = h, v

        # Pre-norm
        h, v = self.norm_h(h), self.norm_v(v)

        # Aggregate messages
        # h -> h
        batch_index = torch.arange(h.shape[0], device=h.device)[:, None, None]
        hij = h[batch_index, neighbor_index]  # [B, N, K, D]
        m_h2h = F.relu(hij + e) * neighbor_mask[..., None]  # [B, N, K, D]
        # v -> h
        m_v2h = F.relu(self.v_to_h(v))[:, None, :]  # [B, 1, D]
        # aggregate messages from neighbors and virtual node
        aggr = m_h2h.sum(dim=-2) + m_v2h  # [B, N, D]

        # h -> v
        m_h2v = F.relu(self.h_to_v(h)) * node_mask[..., None]  # [B, N, D]
        aggr_v = m_h2v.sum(dim=1)  # [B, D]

        # Residual update
        h = _h + self.mlp(h + aggr)
        v = _v + self.mlp(v + aggr_v)

        # Mask out the padded nodes
        h = h * node_mask[..., None]

        return h, v


class MPNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        graph_dim: int,
        model_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.node_dim: int = node_dim
        self.edge_dim: int = edge_dim
        self.graph_dim: int = graph_dim
        self.model_dim: int = model_dim
        self.num_layers: int = num_layers

        self.register_buffer(
            "property_scale", torch.tensor(PROPERTY_SCALE), persistent=False
        )
        # Initialize the node and edge embeddings
        self.node_emb = mlp(node_dim, model_dim, model_dim, 2)
        self.edge_emb = mlp(edge_dim, model_dim, model_dim, 2)

        # Initialize the virtual node with the graph features.
        self.graph_emb = mlp(graph_dim, model_dim, model_dim, 2)

        self.layers = nn.ModuleList([ConvLayer(model_dim) for _ in range(num_layers)])

        self.norm_g = nn.LayerNorm(model_dim)
        self.norm_v = nn.LayerNorm(model_dim)
        self.readout = nn.Linear(2 * model_dim, model_dim)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        h = batch.node_features
        e = batch.edge_features
        node_mask = batch.node_mask.float()
        neighbor_mask = batch.neighbor_mask.float()
        neighbor_index = batch.neighbor_index

        # Initialize node embeddings [B, N, D] and edge embeddings [B, N, K, D].
        h = self.node_emb(h) * node_mask[..., None]  # [B, N, D]
        e = self.edge_emb(e) * neighbor_mask[..., None]  # [B, N, K, D]

        # Initialize one virtual node with the graph features [B, D].
        graph_feat = batch.graph_features / self.property_scale
        v = self.graph_emb(graph_feat)  # [B, D]

        # Update the node and virtual node embeddings
        for layer in self.layers:
            h, v = layer(
                h,
                v,
                e,
                neighbor_index,
                node_mask,
                neighbor_mask,
            )

        # Readout
        g = h.sum(1) / batch.num_nodes.clamp_min(1)[:, None]
        out = self.readout(torch.cat([self.norm_g(g), self.norm_v(v)], dim=1))
        return out
