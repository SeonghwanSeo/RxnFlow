from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rxnflow.envs.features import PROPERTY_SCALE
from rxnflow.envs.graph import GraphBatch

from .nn import mlp


class GINELayer(nn.Module):
    """GINE with node-wise pre-normalization and an additive residual."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.norm_h = nn.LayerNorm(model_dim)
        self.norm_v = nn.LayerNorm(model_dim)
        self.v_to_h = nn.Sequential(nn.Linear(model_dim, model_dim), nn.ReLU())
        self.h_to_v = nn.Sequential(nn.Linear(model_dim, model_dim), nn.ReLU())
        self.mlp = mlp(model_dim, 2 * model_dim, model_dim, 2)

    def forward(
        self,
        h: torch.Tensor,
        v: torch.Tensor,
        e: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update atoms [B, N, D] and virtual nodes [B, D] simultaneously."""
        # Residual connections
        _h, _v = h, v

        # Pre-norm
        h, v = self.norm_h(h), self.norm_v(v)

        # Aggregate messages
        batch_index = torch.arange(h.shape[0], device=h.device)[:, None, None]
        hij = h[batch_index, edge_index]  # [B, N, K, D]
        m = F.relu(hij + e)
        m = m.masked_fill(~edge_mask[..., None], 0)
        aggr = m.sum(dim=2)

        # Exchange messages between atoms and the virtual node.
        # virtual node -> atoms
        aggr = aggr + self.v_to_h(v)[:, None, :]
        # atoms -> virtual node
        aggr_v = self.h_to_v(h).masked_fill(~node_mask[..., None], 0).sum(dim=1)

        # Residual update
        h = _h + self.mlp(h + aggr)
        v = _v + self.mlp(v + aggr_v)

        # Mask out the padded nodes
        h = h.masked_fill(~node_mask[..., None], 0)

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

        self.layers = nn.ModuleList([GINELayer(model_dim) for _ in range(num_layers)])

        self.norm_g = nn.LayerNorm(model_dim)
        self.norm_v = nn.LayerNorm(model_dim)
        self.readout = nn.Linear(2 * model_dim, model_dim)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        # Initialize molecular nodes [B, N, D] and bond embeddings [B, N, K, D].
        h = self.node_emb(batch.node_features)
        e = self.edge_emb(batch.bond_features)
        h = h.masked_fill(~batch.node_mask[..., None], 0)
        e = e.masked_fill(~batch.neighbor_mask[..., None], 0)

        # Initialize one virtual node with the graph features [B, D].
        graph_feat = batch.mol_features / self.property_scale
        v = self.graph_emb(graph_feat)  # [B, D]

        # Update the node and virtual node embeddings
        for layer in self.layers:
            h, v = layer(
                h,
                v,
                e,
                batch.neighbor_index,
                batch.node_mask,
                batch.neighbor_mask,
            )

        # Readout
        g = h.sum(1) / batch.num_nodes.clamp_min(1)[:, None]
        out = self.readout(torch.cat([self.norm_g(g), self.norm_v(v)], dim=1))
        return out
