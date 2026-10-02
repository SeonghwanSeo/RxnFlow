"""Residual GINE message passing with graph conditioning and a virtual node.

The graph encoder uses native Torch, fixed node padding and sparse molecular
messages. Readout concatenates the molecular mean and virtual-node embedding.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rxnflow.envs.graph import GraphBatch


def graph_mlp(n_in: int, hidden: int, n_out: int, layers: int) -> nn.Sequential:
    sizes = [n_in] + [hidden] * layers + [n_out]
    modules = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        # HSX main: activation-aware hidden weights and Xavier linear outputs.
        if i < len(sizes) - 2:
            nn.init.kaiming_uniform_(linear.weight, a=0.01, nonlinearity="leaky_relu")
        else:
            nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        modules.append(linear)
        if i < len(sizes) - 2:
            modules.append(nn.LeakyReLU())
    return nn.Sequential(*modules)


def graph_layer_norm(x: Tensor, valid: Tensor) -> Tensor:
    """Reference graph-mode LayerNorm: normalize all valid node/channel values.

    Tokenwise nn.LayerNorm is a different operation. Virtual nodes count in
    this normalization, whereas padding does not; the reference uses no affine.
    """
    count = valid.sum(1, keepdim=True).clamp_min(1) * x.shape[-1]
    mean = (x * valid[..., None]).sum((1, 2)) / count.squeeze(1)
    centered = (x - mean[:, None, None]) * valid[..., None]
    variance = centered.square().sum((1, 2)) / count.squeeze(1)
    return centered / (variance[:, None, None] + 1e-5).sqrt()


class GINELayer(nn.Module):
    """Pre-normalized GINE with fixed epsilon=0 and a conditioned residual.

    Each layer sums ReLU(h_source + bond) into its target, adds the target's
    own embedding once, then applies a two-Linear update MLP. No explicit
    self-loop edges are needed: the GINE self term already accounts for them.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.update = graph_mlp(hidden, 2 * hidden, hidden, 1)
        self.condition_scale = nn.Linear(hidden, 2 * hidden)

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        valid: Tensor,
        source: Tensor,
        target: Tensor,
        edges: Tensor,
    ) -> Tensor:
        batch, length, hidden = x.shape
        normalized = graph_layer_norm(x, valid).reshape(-1, hidden)
        messages = F.relu(normalized[source] + edges)
        aggregate = torch.zeros_like(normalized).index_add(0, target, messages)
        update = self.update(normalized + aggregate).view(batch, length, hidden)
        # Keep the existing graph-property/capacity/reaction-count conditioning.
        # Reaction identity is still applied later, in the policy heads.
        scale, shift = self.condition_scale(condition).chunk(2, -1)
        return x + update * scale[:, None] + shift[:, None]


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
        self.x2h = graph_mlp(node_dim, hidden_dim, hidden_dim, 2)
        self.e2h = graph_mlp(edge_dim, hidden_dim, hidden_dim, 2)
        # Environment-specific graph condition: properties, remaining capacity,
        # reaction count. Its projection/virtual-node use follows the reference.
        self.c2h = graph_mlp(
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
        valid = torch.cat([batch.node_mask, torch.ones_like(batch.node_mask[:, :1])], 1)
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
            x = layer(x, condition, valid, source, target, edges)
        count = batch.node_mask.sum(1, keepdim=True).clamp_min(1)
        mean = (x[:, :-1] * batch.node_mask[..., None]).sum(1) / count
        # Keep the original 2H readout. No extra learned compression to H.
        return torch.cat([mean, x[:, -1]], -1)
