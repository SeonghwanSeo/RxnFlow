"""Residual GINE message passing with graph conditioning and a virtual node.

The graph encoder uses native Torch and packed molecular nodes/edges.
Readout concatenates the molecular mean and virtual-node embedding.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rxnflow.envs.features import PROPERTY_SCALE
from rxnflow.envs.graph import GraphBatch

from .nn import mlp


class GINELayer(nn.Module):
    """GINE with node-wise pre-normalization and an additive residual.

    Each layer sums ReLU(h_source + bond) into its target, adds the target's
    own embedding once, then applies a two-Linear update MLP. No explicit
    self-loop edges are needed: the GINE self term already accounts for them.
    """

    def __init__(self, model_dim: int):
        super().__init__()
        # Normalize channels within each node, without mixing nodes or padding.
        self.norm = nn.LayerNorm(model_dim)
        self.mlp = mlp(model_dim, 2 * model_dim, model_dim, 2)

    def forward(
        self,
        x: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        edge_emb: torch.Tensor,
    ) -> torch.Tensor:
        """x: [N+B, H], src_idx/dst_idx: [E], edge_emb: [E, H]."""
        node_emb = self.norm(x)
        msg = F.relu(node_emb[src_idx] + edge_emb)
        aggr = torch.zeros_like(node_emb).index_add_(0, dst_idx, msg)
        update = self.mlp(node_emb + aggr)
        return x + update


class MPNN(nn.Module):
    def __init__(
        self,
        x_dim: int,
        e_dim: int,
        g_dim: int,
        cond_dim: int,
        model_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "property_scale", torch.tensor(PROPERTY_SCALE), persistent=False
        )
        self.x2h = mlp(x_dim, model_dim, model_dim, 2)
        self.e2h = mlp(e_dim, model_dim, model_dim, 2)
        # Project molecular features and external conditions separately.
        self.g2h = mlp(g_dim, model_dim, model_dim, 2)
        self.cond2h = nn.Linear(cond_dim, model_dim)
        self.layers = nn.ModuleList([GINELayer(model_dim) for _ in range(num_layers)])
        # Normalize the two readouts separately before concatenating them.
        self.norm_mean = nn.LayerNorm(model_dim)
        self.norm_virtual = nn.LayerNorm(model_dim)

    def forward(self, batch: GraphBatch, cond: torch.Tensor) -> torch.Tensor:
        """cond: [B, cond_dim]; return state readout [B, 2*model_dim]."""
        # 1. Initialize molecular nodes and the graph/condition virtual node.
        node_emb = self.x2h(batch.node_features)
        virtual_emb = self.g2h(batch.mol_features / self.property_scale)
        # External beta/preference conditioning enters once, at initialization.
        virtual_emb = virtual_emb + self.cond2h(cond)
        x = torch.cat([node_emb, virtual_emb], dim=0)
        num_nodes, model_dim = node_emb.shape
        # 2. Build directed bond edges and bidirectional atom/virtual-node edges.
        # Virtual edges use the fixed embedded feature [1, 0, ...].
        src_idx, dst_idx = batch.edge_index
        edge_emb = self.e2h(batch.bond_features)
        atom_indices = torch.arange(num_nodes, device=batch.device)
        virtual_indices = num_nodes + batch.batch_index
        virtual_edges = edge_emb.new_zeros((2 * num_nodes, model_dim))
        virtual_edges[:, 0] = 1
        src_idx = torch.cat([src_idx, atom_indices, virtual_indices])
        dst_idx = torch.cat([dst_idx, virtual_indices, atom_indices])
        edge_emb = torch.cat([edge_emb, virtual_edges])
        # 3. Propagate only over real edges; GINE adds each node's self term.
        for layer in self.layers:
            x = layer(x, src_idx, dst_idx, edge_emb)
        # 4. Pool molecular nodes by graph, excluding the appended virtual nodes.
        mean_emb = x.new_zeros((batch.batch_size, model_dim)).index_add_(
            0, batch.batch_index, x[:num_nodes]
        )
        mean_emb = mean_emb / batch.num_nodes.clamp_min(1)[:, None]
        # Empty states have zero molecular mean before the affine normalization.
        return torch.cat(
            [self.norm_mean(mean_emb), self.norm_virtual(x[num_nodes:])], dim=-1
        )
