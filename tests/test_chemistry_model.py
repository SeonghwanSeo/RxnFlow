from dataclasses import replace

import numpy as np
import pytest
import torch
from model_reference import get_synthon_logits, get_unirxn_logits
from rdkit import Chem

from rxnflow.config import ModelConfig
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.features import heavy_atom_count, parse_molecule
from rxnflow.envs.graph import (
    BOND_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GraphBatch,
    molecule_to_graph_data,
)
from rxnflow.models import RxnFlowModel


def test_heavy_atom_capacity_isotopes_and_no_truncation() -> None:
    empty = molecule_to_graph_data(None, 50)
    assert empty.node_features.shape == (51, NODE_FEATURE_DIM)
    assert empty.bond_features.shape == (51, 51, BOND_FEATURE_DIM)
    assert not empty.node_mask.any()

    explicit_hydrogen = parse_molecule("[H]C([H])([H])[H]")
    assert heavy_atom_count(explicit_hydrogen) == 1
    assert heavy_atom_count(parse_molecule("C[100At]")) == 2

    exact = molecule_to_graph_data(parse_molecule("C" * 50), 50)
    assert exact.node_mask.sum().item() == 50
    with pytest.raises(ValueError, match="exceeding"):
        molecule_to_graph_data(parse_molecule("C" * 51), 50)

    boundary = molecule_to_graph_data(parse_molecule("[1*]" + "C" * 50), 50)
    assert boundary.node_mask.sum().item() == 51

    type_one = molecule_to_graph_data(parse_molecule("[1*]C"), 4)
    type_two = molecule_to_graph_data(parse_molecule("[2*]C"), 4)
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
        env, ModelConfig(num_emb=32, num_layers=2, dropout=0.0), num_objectives=1
    )
    model.eval()
    graph = molecule_to_graph_data(None, env.max_atoms)
    start_embedding = model.graph_embedding(
        GraphBatch.from_graphs([graph]),
        cond_info=_condition(model, len(GraphBatch.from_graphs([graph]).node_mask)),
    )
    assert (
        get_unirxn_logits(
            model,
            start_embedding,
            "nitrile_to_tetrazole",
            logit_scale=model.logit_scale(_condition(model, 1)),
        ).ndim
        == 0
    )

    molecular = molecule_to_graph_data(parse_molecule("CCO"), 12)
    order = torch.tensor([2, 0, 1] + list(range(3, 13)))
    permuted = _permute_graph(molecular, order)
    embeddings = model.graph_embedding(
        GraphBatch.from_graphs([molecular, permuted]),
        cond_info=_condition(
            model, len(GraphBatch.from_graphs([molecular, permuted]).node_mask)
        ),
    )
    assert embeddings.shape == (2, 64)
    assert torch.allclose(embeddings[0], embeddings[1], atol=1e-5)

    no_bonds = replace(molecular, bond_features=torch.zeros_like(molecular.bond_features))
    bond_embeddings = model.graph_embedding(
        GraphBatch.from_graphs([molecular, no_bonds]),
        cond_info=_condition(
            model, len(GraphBatch.from_graphs([molecular, no_bonds]).node_mask)
        ),
    )
    assert not torch.allclose(bond_embeddings[0], bond_embeddings[1])

    library_name = env.brick_types[0]
    indices = torch.arange(min(2, len(env.synthons[library_name])))
    logits = get_synthon_logits(
        model,
        start_embedding,
        "first_synthon",
        library_name,
        indices,
        logit_scale=model.logit_scale(_condition(model, 1)),
    )
    loss = (
        logits.square().mean()
        + get_unirxn_logits(
            model,
            start_embedding,
            "nitrile_to_tetrazole",
            logit_scale=model.logit_scale(_condition(model, 1)),
        ).square()
    )
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_dummy_type_is_categorical_not_molecular_mass() -> None:
    from rxnflow.envs.features import synthon_feature_row

    first, fingerprint_one, _ = synthon_feature_row("[1*]NCC")
    protected, fingerprint_protected, _ = synthon_feature_row("[33*]NCC")
    assert np.array_equal(first, protected)
    # Typed labels still belong to the chemistry features used by the policy.
    assert not np.array_equal(fingerprint_one, fingerprint_protected)

    fingerprints = [synthon_feature_row(f"[{site}*]CC")[1] for site in (0, 1, 2)]
    assert all(fp.dtype == np.uint8 and fp.nbytes == 678 for fp in fingerprints)
    assert len({fp[:512].tobytes() for fp in fingerprints}) == 3


def test_morgan_counts_saturate_before_uint8_conversion() -> None:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    from rxnflow.envs.features import synthon_fingerprint

    # A deliberately long synthetic chain exercises counts above one byte.
    mol = Chem.MolFromSmiles("C" * 300)
    raw = (
        rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
        .GetCountFingerprint(mol)
        .GetNonzeroElements()
    )
    assert max(raw.values()) > 255
    fingerprint = synthon_fingerprint(mol)
    for index, count in raw.items():
        assert int(fingerprint[index]) == min(count, 255)
    assert set(fingerprint[512:]) <= {0, 1}


def test_dot_scores_ignore_synthon_norm_and_train_both_action_scales(
    prepared_env, monkeypatch
) -> None:
    env = SynthesisEnv(prepared_env, max_atoms=20)
    model = RxnFlowModel(env, ModelConfig(num_emb=32, num_layers=1), num_objectives=1)
    state = model.graph_embedding(
        GraphBatch.from_graphs([molecule_to_graph_data(None, 20)]),
        cond_info=_condition(
            model,
            len(GraphBatch.from_graphs([molecule_to_graph_data(None, 20)]).node_mask),
        ),
    )
    library_name = env.brick_types[0]
    indices = torch.arange(2)
    synthons = torch.randn(2, model.emb_type.embedding_dim)
    monkeypatch.setattr("model_reference.get_synthon_emb", lambda *args: synthons)
    before = get_synthon_logits(
        model,
        state,
        "first_synthon",
        library_name,
        indices,
        logit_scale=model.logit_scale(_condition(model, 1)),
    )
    synthons = synthons * torch.tensor([[0.1], [100.0]])
    after = get_synthon_logits(
        model,
        state,
        "first_synthon",
        library_name,
        indices,
        logit_scale=model.logit_scale(_condition(model, 1)),
    )
    assert torch.allclose(before, after, atol=1e-6)
    unary = get_unirxn_logits(
        model,
        state,
        "nitrile_to_tetrazole",
        logit_scale=model.logit_scale(_condition(model, 1)),
    )
    # A single categorical includes both action action_types, so gradients must reach
    # the shared scale and both heads through their shared normalization.
    probability = torch.log_softmax(torch.cat([after, unary.reshape(1)]), dim=0)[-1]
    (-probability).backward()
    assert model._logit_scale[-1].bias.grad[0].abs() > 0
    torch.testing.assert_close(
        model.logit_scale(_condition(model, 1)),
        torch.ones(1),
    )


def test_chirality_and_graph_padding_are_preserved(prepared_env):
    env = SynthesisEnv(prepared_env, max_atoms=12, retrosynthesis_workers=0)
    model = RxnFlowModel(
        env, ModelConfig(num_emb=16, num_layers=2), num_objectives=1
    ).eval()
    graphs = [
        molecule_to_graph_data(parse_molecule(s), 12)
        for s in ("N[C@H](C)O", "N[C@@H](C)O")
    ]
    assert not torch.equal(graphs[0].node_features, graphs[1].node_features)
    assert torch.equal(graphs[0].node_features[:, :-3], graphs[1].node_features[:, :-3])
    batch = GraphBatch.from_graphs(graphs)
    original = model.graph_embedding(
        batch, cond_info=_condition(model, len(batch.node_mask))
    )
    assert not torch.allclose(original[0], original[1])
    # Padding cannot influence message passing or pooling.
    batch.node_features[~batch.node_mask] = 1000
    torch.testing.assert_close(
        model.graph_embedding(batch, cond_info=_condition(model, len(batch.node_mask))),
        original,
    )


def test_graph_encoder_distinguishes_bond_stereoisomers(prepared_env):
    env = SynthesisEnv(prepared_env, max_atoms=12, retrosynthesis_workers=0)
    model = RxnFlowModel(
        env, ModelConfig(num_emb=16, num_layers=2), num_objectives=1
    ).eval()
    # Same atoms and connectivity; only double-bond stereo differs (E/Z/none).
    graphs = [
        molecule_to_graph_data(parse_molecule(s), 12)
        for s in ("F/C=C/F", "F/C=C\\F", "FC=CF")
    ]
    for graph in graphs[1:]:
        assert torch.equal(graphs[0].node_features, graph.node_features)
        assert torch.equal(graphs[0].adjacency, graph.adjacency)
    embeddings = model.graph_embedding(
        GraphBatch.from_graphs(graphs),
        cond_info=_condition(model, len(GraphBatch.from_graphs(graphs).node_mask)),
    )
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert not torch.equal(graphs[i].bond_features, graphs[j].bond_features)
        assert not torch.allclose(embeddings[i], embeddings[j])


def test_gine_layer_matches_equations_and_gradients():
    from copy import deepcopy

    from rxnflow.models.mpnn import GINELayer

    torch.manual_seed(19)
    native = GINELayer(4).double()
    reference = deepcopy(native)
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    # Include an isolated valid node and padding. No self-loop edges: GINE's
    # explicit self term must handle isolated nodes without extra messages.
    valid = torch.tensor([[True, True, True], [True, True, False]])
    source = torch.tensor([0, 1, 3, 4])
    target = torch.tensor([1, 0, 4, 3])
    edges = torch.randn(len(source), 4, dtype=torch.float64, requires_grad=True)
    actual = native(x, source, target, edges)
    rx, re = [v.detach().clone().requires_grad_() for v in (x, edges)]

    # Independent per-destination neighbor sums on normalized node embeddings;
    # do not reuse the native scatter implementation.
    nodes = reference.norm(rx).reshape(-1, 4)
    aggregate = torch.stack(
        [
            nodes[i] + (nodes[source[target == i]] + re[target == i]).relu().sum(0)
            for i in range(6)
        ]
    )
    update = reference.mlp(aggregate).reshape(2, 3, 4)
    expected = rx + update
    torch.testing.assert_close(actual[valid], expected[valid], atol=1e-10, rtol=1e-10)
    actual_grads = torch.autograd.grad(
        actual[valid].square().sum(), (x, edges, *native.parameters())
    )
    expected_grads = torch.autograd.grad(
        expected[valid].square().sum(), (rx, re, *reference.parameters())
    )
    for first, second in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(first, second, atol=1e-9, rtol=1e-9)


def test_mpnn_readout_is_invariant_to_atom_order_and_batch_companions(prepared_env):
    env = SynthesisEnv(prepared_env, max_atoms=12, retrosynthesis_workers=0)
    model = RxnFlowModel(
        env, ModelConfig(num_emb=16, num_layers=2), num_objectives=1
    ).eval()
    mol = parse_molecule("CC(O)N[33*]")
    reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    graphs = [
        molecule_to_graph_data(value, 12)
        for value in (None, mol, reordered, parse_molecule("CCO"))
    ]
    together = model.graph_embedding(
        GraphBatch.from_graphs(graphs),
        cond_info=_condition(model, len(GraphBatch.from_graphs(graphs).node_mask)),
    )
    assert torch.isfinite(together).all()
    torch.testing.assert_close(together[1], together[2], atol=1e-5, rtol=1e-5)
    for i, graph in enumerate(graphs):
        alone = model.graph_embedding(
            GraphBatch.from_graphs([graph]),
            cond_info=_condition(model, len(GraphBatch.from_graphs([graph]).node_mask)),
        )
        torch.testing.assert_close(together[i], alone[0], atol=1e-5, rtol=1e-5)


def _condition(model, count):
    return model.encode_cond(torch.ones(count), torch.ones(count, 1))
