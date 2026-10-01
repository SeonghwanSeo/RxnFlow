from dataclasses import replace

import numpy as np
import pytest
import torch

from rxnflow.config import ModelConfig
from rxnflow.envs.chemistry.features import heavy_atom_count, parse_molecule
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.graph import (
    BOND_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GraphBatch,
    molecule_to_graph_data,
)
from rxnflow.models import RxnFlowModel


def test_heavy_atom_capacity_isotopes_and_no_truncation() -> None:
    empty = molecule_to_graph_data(None, 50, 0)
    assert empty.node_features.shape == (51, NODE_FEATURE_DIM)
    assert empty.bond_features.shape == (51, 51, BOND_FEATURE_DIM)
    assert not empty.node_mask.any()

    explicit_hydrogen = parse_molecule("[H]C([H])([H])[H]")
    assert heavy_atom_count(explicit_hydrogen) == 1
    assert heavy_atom_count(parse_molecule("C[100At]")) == 2

    exact = molecule_to_graph_data(parse_molecule("C" * 50), 50, 1)
    assert exact.node_mask.sum().item() == 50
    with pytest.raises(ValueError, match="exceeding"):
        molecule_to_graph_data(parse_molecule("C" * 51), 50, 1)

    boundary = molecule_to_graph_data(parse_molecule("[1*]" + "C" * 50), 50, 1)
    assert boundary.node_mask.sum().item() == 51

    type_one = molecule_to_graph_data(parse_molecule("[1*]C"), 4, 0)
    type_two = molecule_to_graph_data(parse_molecule("[2*]C"), 4, 0)
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
    graph = molecule_to_graph_data(None, env.max_atoms, 0)
    start_embedding = model.encode_graphs(GraphBatch.from_graphs([graph]))
    assert model.score_scalar(start_embedding, "nitrile_to_tetrazole").ndim == 0

    molecular = molecule_to_graph_data(parse_molecule("CCO"), 12, 1)
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

    fingerprints = [block_feature_row(f"[{site}*]CC")[1] for site in (0, 1, 2)]
    assert all(fp.dtype == np.uint8 and fp.nbytes == 678 for fp in fingerprints)
    assert len({fp[:512].tobytes() for fp in fingerprints}) == 3


def test_morgan_counts_saturate_before_uint8_conversion() -> None:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    from rxnflow.envs.chemistry.features import block_fingerprint

    # A deliberately long synthetic chain exercises counts above one byte.
    mol = Chem.MolFromSmiles("C" * 300)
    raw = (
        rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
        .GetCountFingerprint(mol)
        .GetNonzeroElements()
    )
    assert max(raw.values()) > 255
    fingerprint = block_fingerprint(mol)
    for index, count in raw.items():
        assert int(fingerprint[index]) == min(count, 255)
    assert set(fingerprint[512:]) <= {0, 1}


def test_dot_scores_ignore_block_norm_and_train_both_action_scales(
    prepared_env, monkeypatch
) -> None:
    env = SynthesisEnv(prepared_env, max_atoms=20)
    model = RxnFlowModel(env, ModelConfig(hidden_dim=32, num_heads=4, num_layers=1))
    state = model.encode_graphs(
        GraphBatch.from_graphs([molecule_to_graph_data(None, 20, 0)])
    )
    block_type = env.brick_types[0]
    indices = torch.arange(2)
    blocks = torch.randn(2, 32)
    monkeypatch.setattr(model, "_encode_blocks", lambda *args: blocks)
    before = model.score_blocks(state, "first_block", block_type, indices)
    blocks = blocks * torch.tensor([[0.1], [100.0]])
    after = model.score_blocks(state, "first_block", block_type, indices)
    assert torch.allclose(before, after, atol=1e-6)
    unary = model.score_scalar(state, "nitrile_to_tetrazole")
    # A single categorical includes both action kinds, so gradients must reach
    # both temperatures and both heads through their shared normalization.
    probability = torch.log_softmax(torch.cat([after, unary.reshape(1)]), dim=0)[-1]
    (-probability).backward()
    for name in ("first_block", "nitrile_to_tetrazole"):
        assert model.logit_temperature.grad[env.action_to_index[name]].abs() > 0
    assert torch.allclose(model.temperature, torch.ones_like(model.temperature))


def test_native_attention_matches_explicit_scores_and_gradients() -> None:
    from copy import deepcopy

    from rxnflow.models.graph_transformer import EdgeAwareAttention

    native = EdgeAwareAttention(8, 2, 3, dropout=0).double()
    reference = deepcopy(native)
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    edges = torch.randn(2, 4, 4, 3, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    allowed = valid[:, :, None] & valid[:, None, :]
    allowed.diagonal(dim1=-2, dim2=-1).fill_(True)
    actual = native(x, valid, allowed, edges)
    rx = x.detach().clone().requires_grad_()
    re = edges.detach().clone().requires_grad_()
    q, k, v = reference.qkv(reference.norm(rx)).view(2, 4, 3, 2, 4).unbind(2)
    scores = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1) / 2
    scores = scores + reference.edge_bias(re).permute(0, 3, 1, 2)
    scores = scores.masked_fill(~allowed[:, None], torch.finfo(scores.dtype).min)
    attended = (scores.softmax(-1) @ v.transpose(1, 2)).transpose(1, 2).reshape(2, 4, 8)
    expected = rx + reference.output(attended) * valid[:, :, None]
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(
        actual.square().sum(), (x, edges, *native.parameters())
    )
    expected_grad = torch.autograd.grad(
        expected.square().sum(), (rx, re, *reference.parameters())
    )
    for first, second in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(first, second)
