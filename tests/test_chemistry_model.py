from dataclasses import replace

import pytest
import torch

from rxnflow.chemistry import heavy_atom_count, parse_molecule
from rxnflow.config import ModelConfig
from rxnflow.data.graph import (
    BOND_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GraphBatch,
    molecule_to_graph_data,
)
from rxnflow.envs import SynthesisEnv
from rxnflow.models import RxnFlowModel


def test_heavy_atom_capacity_and_no_truncation() -> None:
    empty = molecule_to_graph_data("", 50, -1, -1, 0)
    assert empty.node_features.shape == (50, NODE_FEATURE_DIM)
    assert empty.bond_features.shape == (50, 50, BOND_FEATURE_DIM)
    assert not empty.node_mask.any()

    explicit_hydrogen = parse_molecule("[H]C([H])([H])[H]")
    assert heavy_atom_count(explicit_hydrogen) == 1
    assert heavy_atom_count(parse_molecule("C[100At]")) == 2

    exact = molecule_to_graph_data("C" * 50, 50, 0, 0, 1)
    assert exact.node_mask.sum().item() == 50
    with pytest.raises(ValueError, match="exceeding"):
        molecule_to_graph_data("C" * 51, 50, 0, 0, 1)


def _permute_graph(graph, permutation: torch.Tensor):
    return replace(
        graph,
        node_features=graph.node_features[permutation],
        node_mask=graph.node_mask[permutation],
        adjacency=graph.adjacency[permutation][:, permutation],
        bond_features=graph.bond_features[permutation][:, permutation],
    )


def test_graph_model_shapes_gradients_permutation_and_bonds(prepared_env) -> None:
    env = SynthesisEnv(prepared_env, max_atoms=12)
    model = RxnFlowModel(
        env, ModelConfig(hidden_dim=32, num_heads=4, num_layers=2, dropout=0.0)
    )
    model.eval()
    state = env.initial_state()
    graph = env.graph_data(state)
    start_embedding = model.encode_graphs(GraphBatch.from_graphs([graph]))
    assert model.score_workflows(start_embedding).shape == (1, 1)

    molecular = molecule_to_graph_data("CCO", 12, 0, 1, 3)
    order = torch.tensor([2, 0, 1] + list(range(3, 12)))
    permuted = _permute_graph(molecular, order)
    embeddings = model.encode_graphs(GraphBatch.from_graphs([molecular, permuted]))
    assert embeddings.shape == (2, 32)
    assert torch.allclose(embeddings[0], embeddings[1], atol=1e-5)

    no_bonds = replace(molecular, bond_features=torch.zeros_like(molecular.bond_features))
    bond_embeddings = model.encode_graphs(GraphBatch.from_graphs([molecular, no_bonds]))
    assert not torch.allclose(bond_embeddings[0], bond_embeddings[1])

    protocol = env.workflows[0].protocols[0]
    indices = torch.tensor([0, 1])
    logits = model.score_blocks(
        start_embedding, protocol.name, protocol.block_type, indices
    )
    loss = logits.square().mean() + model.score_workflows(start_embedding).square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
