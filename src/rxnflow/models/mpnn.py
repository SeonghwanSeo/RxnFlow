"""Residual GINE message passing with graph conditioning and a virtual node.

The graph encoder uses native Torch, fixed node padding and sparse molecular
messages. Readout concatenates the molecular mean and virtual-node embedding.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rxnflow.envs.graph import GraphBatch

from .nn import mlp


class GINELayer(nn.Module):
    """GINE with node-wise pre-normalization and an additive residual.

    Each layer sums ReLU(h_source + bond) into its target, adds the target's
    own embedding once, then applies a two-Linear update MLP. No explicit
    self-loop edges are needed: the GINE self term already accounts for them.
    """

    def __init__(self, hidden: int):
        super().__init__()
        # Normalize channels within each node, without mixing nodes or padding.
        self.norm = nn.LayerNorm(hidden)
        self.update = mlp(hidden, 2 * hidden, hidden, 2)

    def forward(
        self,
        x: Tensor,
        source: Tensor,
        target: Tensor,
        edges: Tensor,
    ) -> Tensor:
        batch, length, hidden = x.shape
        nodes = self.norm(x).reshape(-1, hidden)
        messages = F.relu(nodes[source] + edges)
        aggregate = torch.zeros_like(nodes).index_add(0, target, messages)
        update = self.update(nodes + aggregate).view(batch, length, hidden)
        return x + update


class MPNN(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        mol_feature_dim,
        hidden_dim,
        num_layers,
        max_reactions,
    ):
        super().__init__()
        self.max_reactions = max_reactions
        self.x2h = mlp(node_dim, hidden_dim, hidden_dim, 2)
        self.e2h = mlp(edge_dim, hidden_dim, hidden_dim, 2)
        # Environment-specific graph condition: properties, remaining capacity,
        # reaction count. These enter message passing through the virtual node.
        self.c2h = mlp(
            mol_feature_dim + 1 + max_reactions + 1, hidden_dim, hidden_dim, 2
        )
        self.layers = nn.ModuleList([GINELayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, batch: GraphBatch) -> Tensor:
        nodes = self.x2h(batch.node_features)
        condition = self.c2h(
            torch.cat(
                [
                    batch.mol_features,
                    batch.remaining_capacity[:, None],
                    F.one_hot(batch.reaction_count, self.max_reactions + 1).to(
                        nodes.dtype
                    ),
                ],
                -1,
            )
        )
        x = torch.cat([nodes, condition[:, None]], 1)
        _, length, hidden = x.shape
        # Preserve directed bond order source -> target. Virtual edges have
        # embedded feature [1, 0, ...], as in the original implementation.
        graph, src, dst = batch.adjacency.nonzero(as_tuple=True)
        source, target = graph * length + src, graph * length + dst
        edges = self.e2h(batch.bond_features[graph, src, dst])
        graph, atom = batch.node_mask.nonzero(as_tuple=True)
        atom_indices = graph * length + atom
        virtual_indices = graph * length + length - 1
        virtual_edges = edges.new_zeros((2 * len(atom), hidden))
        virtual_edges[:, 0] = 1
        source = torch.cat([source, atom_indices, virtual_indices])
        target = torch.cat([target, virtual_indices, atom_indices])
        edges = torch.cat([edges, virtual_edges])
        # GINE includes its own self term; use only bond and virtual edges.
        for layer in self.layers:
            x = layer(x, source, target, edges)
        count = batch.node_mask.sum(1, keepdim=True).clamp_min(1)
        mean = (x[:, :-1] * batch.node_mask[..., None]).sum(1) / count
        # Keep the original 2H readout. No extra learned compression to H.
        return torch.cat([mean, x[:, -1]], -1)
