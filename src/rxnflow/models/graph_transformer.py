"""RxnFlow/HSX graph-transformer equations using native Torch operations.

Graph inputs remain padded. Message passing uses molecular edges, virtual-node
edges and self loops; it never builds a dense [node, node, head, hidden] tensor.
The baseline is pre-norm GENConv(add) + TransformerConv(concat heads, root skip).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rxnflow.envs.graph import GraphBatch


def graph_mlp(n_in: int, hidden: int, n_out: int, layers: int) -> nn.Sequential:
    sizes = [n_in] + [hidden] * layers + [n_out]
    modules = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        # Reference GraphTransformer.reset_parameters applies this to its MLPs.
        nn.init.kaiming_uniform_(linear.weight, a=0.01, nonlinearity="leaky_relu")
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


class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        # Reference GENConv(num_layers=1, norm=None, bias=False) is one linear
        # after sum(ReLU(x_j + edge) + 1e-7) + x_i.
        self.gen = nn.Linear(hidden, hidden, bias=False)
        # concat_heads=True: EACH head has H channels, not H / heads.
        self.query = nn.Linear(2 * hidden, heads * hidden)
        self.key = nn.Linear(2 * hidden, heads * hidden)
        self.value = nn.Linear(2 * hidden, heads * hidden)
        self.edge = nn.Linear(hidden, heads * hidden, bias=False)
        self.skip = nn.Linear(2 * hidden, heads * hidden)
        self.output = nn.Linear(heads * hidden, hidden)
        self.condition_scale = nn.Linear(hidden, 2 * hidden)
        # The reference conv/linear modules use their default Kaiming-uniform
        # initialization; only the graph MLPs use the explicit LeakyReLU rule.
        self.ff = graph_mlp(hidden, 4 * hidden, hidden, 1)

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
        messages = F.relu(normalized[source] + edges) + 1e-7
        aggregate = torch.zeros_like(normalized).index_add(0, target, messages)
        aggregate = self.gen(aggregate + normalized)
        joined = torch.cat([normalized, aggregate], -1)
        q = self.query(joined).view(-1, self.heads, hidden)
        k = self.key(joined).view(-1, self.heads, hidden)
        v = self.value(joined).view(-1, self.heads, hidden)
        edge = self.edge(edges).view(-1, self.heads, hidden)
        scores = (q[target] * (k[source] + edge)).sum(-1) / math.sqrt(hidden)
        # Native grouped softmax on incoming molecular edges. Detaching the
        # stabilizing maximum matches the reference and avoids max gradients.
        indices = target[:, None].expand(-1, self.heads)
        maxima = scores.new_full((batch * length, self.heads), -torch.inf)
        maxima.scatter_reduce_(
            0, indices, scores.detach(), reduce="amax", include_self=True
        )
        exponent = (scores - maxima[target]).exp()
        totals = torch.zeros_like(maxima).index_add(0, target, exponent)
        alpha = exponent / (totals[target] + 1e-16)
        values = (v[source] + edge) * alpha[..., None]
        attended = v.new_zeros(v.shape).index_add(0, target, values).flatten(1)
        update = self.output(attended + self.skip(joined)).view(batch, length, hidden)
        scale, shift = self.condition_scale(condition).chunk(2, -1)
        x = x + update * scale[:, None] + shift[:, None]
        return x + self.ff(graph_layer_norm(x, valid))


class GraphTransformer(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        mol_feature_dim,
        hidden_dim,
        num_heads,
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
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )

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
        n_batch, length, hidden = x.shape
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
        # add_self_loops(fill_value='mean'): mean of incoming edge embeddings,
        # including virtual edges. Isolated virtual nodes receive zero.
        sums = edges.new_zeros((n_batch * length, hidden)).index_add(0, target, edges)
        counts = edges.new_zeros(n_batch * length).index_add(
            0, target, edges.new_ones(len(target))
        )
        loops = valid.flatten().nonzero().flatten()
        edges = torch.cat([edges, sums[loops] / counts[loops, None].clamp_min(1)])
        source, target = torch.cat([source, loops]), torch.cat([target, loops])
        for layer in self.layers:
            x = layer(x, condition, valid, source, target, edges)
        count = batch.node_mask.sum(1, keepdim=True).clamp_min(1)
        mean = (x[:, :-1] * batch.node_mask[..., None]).sum(1) / count
        # Keep the original 2H readout. No extra learned compression to H.
        return torch.cat([mean, x[:, -1]], -1)
