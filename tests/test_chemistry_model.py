from dataclasses import replace

import numpy as np
import pytest
import torch

from rxnflow.config import ModelConfig
from rxnflow.envs import SynthesisEnv
from rxnflow.envs.chemistry.features import heavy_atom_count, parse_molecule
from rxnflow.envs.graph import (
    BOND_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GraphBatch,
    molecule_to_graph_data,
)
from rxnflow.models import RxnFlowModel


def test_heavy_atom_capacity_isotopes_and_no_truncation() -> None:
    empty = molecule_to_graph_data("", 50, 0)
    assert empty.node_features.shape == (51, NODE_FEATURE_DIM)
    assert empty.bond_features.shape == (51, 51, BOND_FEATURE_DIM)
    assert not empty.node_mask.any()

    explicit_hydrogen = parse_molecule("[H]C([H])([H])[H]")
    assert heavy_atom_count(explicit_hydrogen) == 1
    assert heavy_atom_count(parse_molecule("C[100At]")) == 2

    exact = molecule_to_graph_data("C" * 50, 50, 1)
    assert exact.node_mask.sum().item() == 50
    with pytest.raises(ValueError, match="exceeding"):
        molecule_to_graph_data("C" * 51, 50, 1)

    boundary = molecule_to_graph_data("[1*]" + "C" * 50, 50, 1)
    assert boundary.node_mask.sum().item() == 51

    type_one = molecule_to_graph_data("[1*]C", 4, 0)
    type_two = molecule_to_graph_data("[2*]C", 4, 0)
    assert not torch.equal(type_one.node_features[0], type_two.node_features[0])


def _permute_graph(graph, permutation: torch.Tensor):
    return replace(
        graph,
        node_features=graph.node_features[permutation],
        node_mask=graph.node_mask[permutation],
        adjacency=graph.adjacency[permutation][:, permutation],
        bond_features=graph.bond_features[permutation][:, permutation],
    )


def test_graph_model_shapes_gradients_permutation_and_bonds(prepared_env) -> None:
    env = SynthesisEnv(prepared_env, max_atoms=12, retrosynthesis_workers=0)
    model = RxnFlowModel(
        env, ModelConfig(hidden_dim=32, num_heads=4, num_layers=2, dropout=0.0)
    )
    model.eval()
    graph = molecule_to_graph_data("", env.max_atoms, 0)
    start_embedding = model.encode_graphs(GraphBatch.from_graphs([graph]))
    assert model.score_scalar(start_embedding, "nitrile_to_tetrazole").ndim == 0

    molecular = molecule_to_graph_data("CCO", 12, 1)
    order = torch.tensor([2, 0, 1] + list(range(3, 13)))
    permuted = _permute_graph(molecular, order)
    embeddings = model.encode_graphs(GraphBatch.from_graphs([molecular, permuted]))
    assert embeddings.shape == (2, 32)
    assert torch.allclose(embeddings[0], embeddings[1], atol=1e-5)

    no_bonds = replace(molecular, bond_features=torch.zeros_like(molecular.bond_features))
    bond_embeddings = model.encode_graphs(GraphBatch.from_graphs([molecular, no_bonds]))
    assert not torch.allclose(bond_embeddings[0], bond_embeddings[1])

    block_type = env.brick_types[0]
    indices = torch.arange(min(2, len(env.blocks[block_type])))
    logits = model.score_blocks(start_embedding, "first_block", block_type, indices)
    loss = (
        logits.square().mean()
        + model.score_scalar(start_embedding, "nitrile_to_tetrazole").square()
    )
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_dummy_type_is_categorical_not_molecular_mass() -> None:
    from rxnflow.envs.chemistry.features import block_feature_row

    first, fingerprint_one, _ = block_feature_row("[1*]NCC")
    protected, fingerprint_protected, _ = block_feature_row("[33*]NCC")
    assert np.array_equal(first, protected)
    # Typed labels still belong to the chemistry features used by the policy.
    assert not np.array_equal(fingerprint_one, fingerprint_protected)
