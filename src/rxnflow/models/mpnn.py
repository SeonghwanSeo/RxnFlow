"""Residual GINE message passing with graph conditioning and a virtual node.

The graph encoder uses native Torch, fixed node padding and sparse molecular
messages. Readout concatenates the molecular mean and virtual-node embedding.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rxnflow.envs.graph import GraphBatch

from .nn import mlp


class GINELayer(nn.Module):
    """GINE with node-wise pre-normalization and an additive residual.

    Each layer sums ReLU(h_source + bond) into its target, adds the target's
    own embedding once, then applies a two-Linear update MLP. No explicit
    self-loop edges are needed: the GINE self term already accounts for them.
    """

    def __init__(self, num_emb: int):
        super().__init__()
        # Normalize channels within each node, without mixing nodes or padding.
        self.norm = nn.LayerNorm(num_emb)
        self.mlp = mlp(num_emb, 2 * num_emb, num_emb, 2)

    def forward(
        self,
        x: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        edge_emb: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, num_emb = x.shape
        node_emb = self.norm(x).reshape(-1, num_emb)
        msg = F.relu(node_emb[src_idx] + edge_emb)
        aggr = torch.zeros_like(node_emb).index_add(0, dst_idx, msg)
        update = self.mlp(node_emb + aggr).view(batch, length, num_emb)
        return x + update


class MPNN(nn.Module):
    def __init__(
        self,
        x_dim,
        e_dim,
        g_dim,
        num_emb,
        num_layers,
        max_reactions,
    ):
        super().__init__()
        self.max_reactions = max_reactions
        self.x2h = mlp(x_dim, num_emb, num_emb, 2)
        self.e2h = mlp(e_dim, num_emb, num_emb, 2)
        # Environment-specific graph condition: properties, remaining capacity,
        # reaction count. These enter message passing through the virtual node.
        self.c2h = mlp(g_dim + 1 + max_reactions + 1, num_emb, num_emb, 2)
        self.layers = nn.ModuleList([GINELayer(num_emb) for _ in range(num_layers)])

    def forward(self, batch: GraphBatch, cond_info: torch.Tensor) -> torch.Tensor:
        node_emb = self.x2h(batch.node_features)
        virtual_emb = self.c2h(
            torch.cat(
                [
                    batch.mol_features,
                    batch.remaining_capacity[:, None],
                    F.one_hot(batch.reaction_count, self.max_reactions + 1).to(
                        node_emb.dtype
                    ),
                ],
                -1,
            )
        )
        # External beta/preference conditioning enters once, at initialization.
        virtual_emb = virtual_emb + cond_info
        x = torch.cat([node_emb, virtual_emb[:, None]], 1)
        _, length, num_emb = x.shape
        # Preserve directed bond order source -> target. Virtual edges have
        # embedded feature [1, 0, ...], as in the original implementation.
        graph, src, dst = batch.adjacency.nonzero(as_tuple=True)
        src_idx, dst_idx = graph * length + src, graph * length + dst
        edge_emb = self.e2h(batch.bond_features[graph, src, dst])
        graph, atom = batch.node_mask.nonzero(as_tuple=True)
        atom_indices = graph * length + atom
        virtual_indices = graph * length + length - 1
        virtual_edges = edge_emb.new_zeros((2 * len(atom), num_emb))
        virtual_edges[:, 0] = 1
        src_idx = torch.cat([src_idx, atom_indices, virtual_indices])
        dst_idx = torch.cat([dst_idx, virtual_indices, atom_indices])
        edge_emb = torch.cat([edge_emb, virtual_edges])
        # GINE includes its own self term; use only bond and virtual edges.
        for layer in self.layers:
            x = layer(x, src_idx, dst_idx, edge_emb)
        count = batch.node_mask.sum(1, keepdim=True).clamp_min(1)
        mean_emb = (x[:, :-1] * batch.node_mask[..., None]).sum(1) / count
        # Keep the original 2H readout. No extra learned compression to H.
        return torch.cat([mean_emb, x[:, -1]], -1)
