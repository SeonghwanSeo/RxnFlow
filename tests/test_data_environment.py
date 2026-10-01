import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from rxnflow.envs import SynthesisEnv
from rxnflow.envs.prepare import convert_stage, features_stage
from rxnflow.types import ActionKind, MoleculeState


def block_index(env, block_type, smiles):
    return env.blocks[block_type].smiles.index(
        Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    )


def actions_for(env, state, name, block_type=None, smiles=None):
    group = next(
        group
        for group in env.available_groups(state)
        if group.name == name and group.block_type == block_type
    )
    index = block_index(env, block_type, smiles) if smiles is not None else None
    return env.outcomes(state, group, index)


def test_pipeline_is_aligned_and_preserves_sources(prepared_env: Path) -> None:
    required = {
        "synthon.yaml",
        "reaction.yaml",
        "building_blocks.json",
        "bb_feature.npz",
        "prepare_manifest.json",
    }
    assert required <= {path.name for path in prepared_env.iterdir()}
    block_names = {path.name for path in (prepared_env / "blocks").glob("*.smi")}
    assert {"1.smi", "1-1.smi", "1-3.smi", "3-33.smi", "34.smi", "35.smi"} <= block_names
    assert not list(prepared_env.rglob("*cluster*"))
    from rxnflow.cli.prepare import main

    root = Path(__file__).parents[1]
    with pytest.raises(SystemExit, match="already exists"):
        main(
            [
                "--env-dir",
                str(prepared_env),
                "--building-blocks",
                str(root / "tests/fixtures/enamine_stock.smi"),
                "--template-dir",
                str(root / "data/templates"),
            ]
        )
    env = SynthesisEnv(prepared_env, max_atoms=30)
    assert len(env.bi_reactions) == 38
    assert len(env.uni_reactions) == 4
    assert len(env.synthon_types) == 35
    i = block_index(env, "1", "[1*]NCCN")
    assert env.blocks["1"].identifiers[i] == ["EN-A", "EN-A2"]
    assert env.sources["EN-A"] == "NCCN"
    # One primary amine cannot be counted twice as a two-site linker.
    assert "[1*]N([1*])CCN" not in env.blocks["1-1"].smiles
    assert all(len(row) == 9 for row in env.blocks["1"].properties)


def test_coupling_deprotection_coupling_and_provenance(prepared_env: Path) -> None:
    env = SynthesisEnv(prepared_env, max_atoms=30, max_reactions=3)
    initial = env.initial_state()
    assert {g.kind for g in env.available_groups(initial)} == {ActionKind.FIRST_BLOCK}
    first = actions_for(env, initial, "first_block", "1", "[1*]NCCN")[0]
    start = env.step(initial, first)
    assert start.reaction_count == 0
    coupling = actions_for(env, start, "rxn1_b1", "3-33", "[3*]CN[33*]")[0]
    protected = env.step(start, coupling)
    assert env.dummy_signature(protected.smiles) == (33,)
    assert {g.kind for g in env.available_groups(protected)} == {ActionKind.UNI_REACTION}
    deprotect = actions_for(env, protected, "boc_deprotection")[0]
    activated = env.step(protected, deprotect)
    assert activated.reaction_count == 2
    assert env.dummy_signature(activated.smiles) == (1,)
    closure = actions_for(env, activated, "rxn1_b1", "3", "C[3*]")[0]
    terminal = env.step(activated, closure)
    assert terminal.terminated and terminal.reaction_count == 3
    assert "*" not in terminal.smiles and not env.available_groups(terminal)
    assert all(env.blocks[g.block_type].is_brick for g in env.available_groups(activated))
    public = env.action_to_dict(first)
    assert public["block_ids"] == ["EN-A", "EN-A2"]
    assert public["building_blocks"][0] == {"id": "EN-A", "smiles": "NCCN"}
    assert env.action_to_dict(coupling)["block_role"] == "linker"
    assert env.backward_log_probability(start, first, "") is not None
    assert (
        env.backward_log_probability(activated, deprotect, protected.smiles) is not None
    )


def test_terminal_unary_early_exit_and_final_step_masks(prepared_env: Path) -> None:
    env = SynthesisEnv(prepared_env, max_atoms=30, max_reactions=3)
    first = actions_for(env, env.initial_state(), "first_block", "11", "CC[11*]")[0]
    state = env.step(env.initial_state(), first)
    action = actions_for(env, state, "nitrile_to_tetrazole")[0]
    early = env.step(state, action)
    assert early.terminated and early.reaction_count == 1
    assert early.smiles == "CCc1nnn[nH]1"
    last = replace(state, reaction_count=2)
    assert actions_for(env, last, "nitrile_to_tetrazole") == [action]
    assert env.step(last, action).terminated
    assert env.available_groups(replace(state, reaction_count=3)) == []
    assert env.available_groups(MoleculeState("[33*]NCC", reaction_count=2)) == []
    assert not env.available_groups(early)
    with pytest.raises(ValueError, match="not available"):
        env.step(early, action)


def test_non_equivalent_linker_sites_are_separate_actions(prepared_env: Path) -> None:
    env = SynthesisEnv(prepared_env, max_atoms=30, max_reactions=3)
    state = MoleculeState("[3*]C")
    outcomes = actions_for(env, state, "rxn1_b0", "1-1", "[1*]NCCC(C)N[1*]")
    assert len(outcomes) == 2
    assert len({action.product_smiles for action in outcomes}) == 2
    for action in outcomes:
        product = env.step(state, action)
        assert env.dummy_signature(product.smiles) == (1,)
        assert not product.terminated
    # Symmetry-equivalent terminal amines yield one product, not two actions.
    assert len(actions_for(env, state, "rxn1_b0", "1-1", "[1*]NCCN[1*]")) == 1


def test_exact_product_limits_include_inserted_atoms_and_unary(
    prepared_env: Path,
) -> None:
    # Amidation inserts C=O: the two synthon heavy-atom counts alone undercount.
    env = SynthesisEnv(prepared_env, max_atoms=5)
    state = MoleculeState("[1*]NCCN")
    assert actions_for(env, state, "rxn1_b1", "3", "C[3*]") == []
    relaxed = SynthesisEnv(prepared_env, max_atoms=7)
    assert actions_for(relaxed, state, "rxn1_b1", "3", "C[3*]")
    # Terminal tetrazole is seven heavy atoms; it must respect the same limits.
    assert actions_for(env, MoleculeState("[11*]CC"), "nitrile_to_tetrazole") == []
    property_limited = SynthesisEnv(
        prepared_env, max_atoms=30, property_penalty={"rings": 0.5}
    )
    assert (
        actions_for(property_limited, MoleculeState("[11*]CC"), "nitrile_to_tetrazole")
        == []
    )


def test_reconversion_invalidates_features(prepared_env: Path, tmp_path: Path) -> None:

    root = Path(__file__).parents[1]
    env_dir = tmp_path / "rebuild"
    shutil.copytree(prepared_env, env_dir)
    convert_stage(
        root / "tests/fixtures/enamine_stock.smi",
        env_dir,
        root / "data/templates",
    )
    assert not (env_dir / "bb_feature.npz").exists()
    assert (
        "features"
        not in json.loads((env_dir / "prepare_manifest.json").read_text())["stages"]
    )
    features_stage(env_dir)
    assert SynthesisEnv(env_dir).blocks


def test_versioned_feature_artifact_is_rejected(
    prepared_env: Path, tmp_path: Path
) -> None:
    env_dir = tmp_path / "old"
    shutil.copytree(prepared_env, env_dir)
    path = env_dir / "bb_feature.npz"
    with np.load(path) as arrays:
        data = dict(arrays)
    data["format"] = np.array("rxnflow-bb-feature-v1")
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="unsupported bb_feature.npz"):
        SynthesisEnv(env_dir)


def test_minimum_reactions_masks_early_termination(prepared_env: Path) -> None:
    env = SynthesisEnv(prepared_env, min_reactions=2, max_reactions=3)
    state = MoleculeState("[3*]C")
    assert env.available_groups(state)
    assert all(not env.blocks[g.block_type].is_brick for g in env.available_groups(state))
    assert "nitrile_to_tetrazole" not in {
        g.name for g in env.available_groups(MoleculeState("[11*]CC"))
    }
    allowed = MoleculeState("[11*]CC", reaction_count=1)
    assert env.step(
        allowed, actions_for(env, allowed, "nitrile_to_tetrazole")[0]
    ).terminated
