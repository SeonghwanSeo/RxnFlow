import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from rxnflow.core.errors import InvalidTransition
from rxnflow.core.types import Action, ActionSubspace, ActionType, State
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.graph import molecule_to_graph_data
from rxnflow.envs.prepare import convert_stage, features_stage


def block_index(env, block_type, smiles):
    return env.blocks[block_type].smiles.index(
        Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    )


def actions_for(env, state, name, block_type=None, smiles=None):
    subspace = next(
        subspace
        for subspace in env.get_action_space(state)
        if subspace.name == (name, block_type)
    )
    index = block_index(env, block_type, smiles) if smiles is not None else None
    action = Action(
        subspace.action_type,
        None if subspace.action_type == ActionType.FIRST_BLOCK else subspace.name[0],
        block_type,
        index,
    )
    try:
        env.step(state, action)
    except InvalidTransition:
        return []
    return [action]


def test_pipeline_is_aligned_and_preserves_sources(prepared_env: Path) -> None:
    required = {
        "synthon.yaml",
        "reaction.yaml",
        "building_blocks.json",
        "bb_feature.npz",
        "prepare_manifest.json",
        "action_space.json",
        "signature.json",
    }
    assert required <= {path.name for path in prepared_env.iterdir()}
    block_names = {path.name for path in (prepared_env / "blocks").glob("*.smi")}
    assert {"1.smi", "1-1.smi", "1-3.smi", "3-33.smi", "34.smi", "35.smi"} <= block_names
    assert not list(prepared_env.rglob("*cluster*"))
    from scripts.prepare import main

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
    with np.load(prepared_env / "bb_feature.npz") as arrays:
        for name, library in env.blocks.items():
            assert arrays[f"{name}/fingerprints"].dtype == np.uint8
            assert library.fingerprints.dtype == np.uint8
            assert isinstance(library.properties, np.ndarray)
            assert library.properties.dtype == np.float32
            assert library.heavy_atoms.dtype == np.uint8
            assert arrays[f"{name}/heavy_atoms"].dtype == np.uint8
            np.testing.assert_array_equal(library.heavy_atoms, library.properties[:, -1])
    assert len({name.rsplit("_", 2)[0] for name in env.bi_reactions}) == 38
    assert len(env.uni_reactions) == 4
    assert len(env.synthon_types) == 35
    i = block_index(env, "1", "*NCCN")
    assert env.blocks["1"].identifiers[i] == ["EN-A", "EN-A2"]
    assert env.sources["EN-A"] == "NCCN"
    # One primary amine cannot be counted twice as a two-site linker.
    assert "*N([1*])CCN" not in env.blocks["1-1"].smiles
    assert all(len(row) == 9 for row in env.blocks["1"].properties)


def test_coupling_deprotection_coupling_and_provenance(prepared_env: Path) -> None:
    env = SynthesisEnv(prepared_env, max_atoms=30, max_reactions=3)
    initial = env.initial_state()
    assert env.get_action_space(initial) is env.initial_action_space
    assert set(env.reaction_action_spaces) == env.synthon_types
    # Step variants are precomputed; equivalent states reuse their subspace list.
    same_site = State.from_smiles("[1*]NCC")
    assert env.get_action_space(same_site) is env.get_action_space(same_site)
    assert all(
        isinstance(subspace, ActionSubspace) for subspace in env.reaction_action_spaces[1]
    )
    assert {g.action_type for g in env.get_action_space(initial)} == {
        ActionType.FIRST_BLOCK
    }
    first = actions_for(env, initial, "first_block", "1", "*NCCN")[0]
    start = env.step(initial, first)
    assert start.reaction_count == 0
    coupling = actions_for(env, start, "amide_coupling_state_first", "3-33", "*CN[33*]")[
        0
    ]
    protected = env.step(start, coupling)
    assert env.get_synthon_types(protected.smiles) == (33,)
    assert {g.action_type for g in env.get_action_space(protected)} == {
        ActionType.UNI_REACTION
    }
    deprotect = actions_for(env, protected, "boc_deprotection")[0]
    activated = env.step(protected, deprotect)
    assert activated.reaction_count == 2
    assert env.get_synthon_types(activated.smiles) == (1,)
    closure = actions_for(env, activated, "amide_coupling_state_first", "3", "C*")[0]
    terminal = env.step(activated, closure)
    assert terminal.terminated and terminal.reaction_count == 3
    assert "*" not in terminal.smiles and not env.get_action_space(terminal)
    assert all(env.blocks[g.name[1]].is_brick for g in env.get_action_space(activated))
    public = env.action_to_dict(first)
    assert public["block_ids"] == ["EN-A", "EN-A2"]
    assert public["building_blocks"][0] == {"id": "EN-A", "smiles": "NCCN"}
    assert env.action_to_dict(coupling)["block_type"] == "linker"
    assert any(
        route[0] == (first, "") for route in env.retro_analyzer.run(start.smiles, 0)
    )
    assert any(
        route[0] == (deprotect, protected.smiles)
        for route in env.retro_analyzer.run(activated.smiles, activated.reaction_count)
    )


def test_state_retains_product_molecule_and_stereochemistry(
    prepared_env: Path,
    monkeypatch,
) -> None:
    env = SynthesisEnv(prepared_env, max_atoms=30)
    parent = State.from_smiles("[1*]N[C@@H](C)C/C=C/c1ccc([N+](=O)[O-])cc1")
    parent_smiles = parent.smiles
    expected = Chem.MolToSmiles(
        Chem.MolFromSmiles("CC(=O)N[C@@H](C)C/C=C/c1ccc([N+](=O)[O-])cc1")
    )
    action = actions_for(env, parent, "amide_coupling_state_first", "3", "C*")[0]

    # The selected block is stored as SMILES, but the parent and product must
    # stay as molecule objects throughout the transition and graph encoding.
    block_smiles = env.blocks[action.block_type].smiles[action.block_index]
    parse = Chem.MolFromSmiles
    parsed = []

    def parse_block(smiles, *args, **kwargs):
        assert smiles == block_smiles, "state/graph encoding reparsed a molecule"
        parsed.append(smiles)
        return parse(smiles, *args, **kwargs)

    monkeypatch.setattr(Chem, "MolFromSmiles", parse_block)
    child = env.step(parent, action)
    graph = molecule_to_graph_data(child.mol, env.max_atoms, child.reaction_count)
    restored = pickle.loads(pickle.dumps(child))
    restored_graph = molecule_to_graph_data(
        restored.mol, env.max_atoms, restored.reaction_count
    )
    assert parsed == [block_smiles]
    assert child.smiles == expected
    assert restored.smiles == expected == Chem.MolToSmiles(restored.mol)
    assert child.mol is not parent.mol
    assert Chem.MolToSmiles(parent.mol) == parent_smiles
    assert np.array_equal(
        graph.node_features.numpy(), restored_graph.node_features.numpy()
    )
    assert np.array_equal(
        graph.bond_features.numpy(), restored_graph.bond_features.numpy()
    )


def test_terminal_unary_early_exit_and_final_step_masks(prepared_env: Path) -> None:
    env = SynthesisEnv(prepared_env, max_atoms=30, max_reactions=3)
    first = actions_for(env, env.initial_state(), "first_block", "11", "CC*")[0]
    state = env.step(env.initial_state(), first)
    action = actions_for(env, state, "nitrile_to_tetrazole")[0]
    early = env.step(state, action)
    assert early.terminated and early.reaction_count == 1
    assert early.smiles == "CCc1nnn[nH]1"
    last = replace(state, reaction_count=2)
    assert actions_for(env, last, "nitrile_to_tetrazole") == [action]
    assert env.step(last, action).terminated
    assert env.get_action_space(replace(state, reaction_count=3)) == []
    assert env.get_action_space(State.from_smiles("[33*]NCC", reaction_count=2)) == []
    assert not env.get_action_space(early)
    with pytest.raises(InvalidTransition, match="structural or graph-capacity"):
        env.step(early, action)


def test_linker_orientation_fixes_attachment_and_reverse_catalog_lookup(
    prepared_env: Path,
) -> None:
    env = SynthesisEnv(prepared_env, max_atoms=30, max_reactions=3)
    state = State.from_smiles("[3*]C")
    directions = ["*NCCC(C)N[1*]", "[1*]NCCC(C)N*"]
    outcomes = []
    products = []
    fingerprints = []
    for smiles in directions:
        actions = actions_for(env, state, "amide_coupling_block_first", "1-1", smiles)
        assert len(actions) == 1
        action = actions[0]
        outcomes.append(action)
        fingerprints.append(env.blocks["1-1"].fingerprints[action.block_index])
        product = env.step(state, action)
        products.append(product.smiles)
        assert env.get_synthon_types(product.smiles) == (1,)
        assert not product.terminated
        assert any(
            route[0] == (action, state.smiles)
            for route in env.retro_analyzer.run(product.smiles, product.reaction_count)
        )
    assert outcomes[0].block_index != outcomes[1].block_index
    assert products[0] != products[1]
    assert not np.array_equal(fingerprints[0], fingerprints[1])
    # Equivalent orientations collapse to one catalog row.
    symmetric = Chem.MolToSmiles(Chem.MolFromSmiles("*NCCN[1*]"))
    assert env.blocks["1-1"].smiles.count(symmetric) == 1
    assert (
        len(actions_for(env, state, "amide_coupling_block_first", "1-1", symmetric)) == 1
    )

    # Ordered library types determine which end attaches, even for two types.
    assert "1-3" in env.blocks and "3-1" in env.blocks
    assert all(
        g.name[1] != "1-3"
        for g in env.get_action_space(State.from_smiles("[1*]NCC"))
        if g.name[0] == "amide_coupling_state_first"
    )


def test_selected_product_capacity_and_estimated_property_budget(
    prepared_env: Path,
) -> None:
    # Amidation inserts C=O: the two synthon heavy-atom counts alone undercount.
    env = SynthesisEnv(prepared_env, max_atoms=5)
    state = State.from_smiles("[1*]NCCN")
    assert actions_for(env, state, "amide_coupling_state_first", "3", "C*") == []
    relaxed = SynthesisEnv(prepared_env, max_atoms=7)
    assert actions_for(relaxed, state, "amide_coupling_state_first", "3", "C*")
    # Terminal tetrazole is seven heavy atoms; it must respect the same limits.
    assert actions_for(env, State.from_smiles("[11*]CC"), "nitrile_to_tetrazole") == []
    property_limited = SynthesisEnv(
        prepared_env, max_atoms=30, property_penalty={"rings": 0}
    )
    # Property masking is a reactant-budget estimate, not an exact product
    # constraint. Unary ring formation is not vetoed by a product descriptor.
    assert actions_for(
        property_limited, State.from_smiles("[11*]CC"), "nitrile_to_tetrazole"
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
    assert not (env_dir / "action_space.json").exists()
    assert not (env_dir / "signature.json").exists()
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


def test_regular_and_last_action_spaces(prepared_env: Path) -> None:
    env = SynthesisEnv(prepared_env, max_reactions=3)
    state = State.from_smiles("[3*]C")
    assert env.get_action_space(state) is env.reaction_action_spaces[3]
    libraries = [
        subspace.name[1]
        for subspace in env.get_action_space(state)
        if subspace.name[1] is not None
    ]
    assert any(env.blocks[name].is_brick for name in libraries)
    assert any(not env.blocks[name].is_brick for name in libraries)
    last = replace(state, reaction_count=2)
    assert env.get_action_space(last) is env.last_action_spaces[3]
    assert all(
        env.blocks[subspace.name[1]].is_brick
        for subspace in env.get_action_space(last)
        if subspace.name[1] is not None
    )
    assert "nitrile_to_tetrazole" in {
        subspace.name[0]
        for subspace in env.get_action_space(State.from_smiles("[11*]CC"))
    }
    # With one allowed reaction, the first post-FirstBlock state uses last space.
    single = SynthesisEnv(prepared_env, max_reactions=1)
    assert single.get_action_space(state) is single.last_action_spaces[3]


def test_budget_tolerance_and_nonpositive_bounds(prepared_env: Path) -> None:
    from rxnflow.envs.features import PROPERTY_DIM, PROPERTY_NAMES

    env = SynthesisEnv(
        prepared_env, property_penalty={"mw": 100.0, "rings": 0, "logp": -1.0}
    )
    name = env.brick_types[0]
    library = env.blocks[name]
    library.properties[:, PROPERTY_NAMES.index("mw")] = 100.5
    library.properties[:, PROPERTY_NAMES.index("rings")] = 0
    library.properties[:, PROPERTY_NAMES.index("logp")] = -1.0
    state_properties = np.zeros(PROPERTY_DIM, dtype=np.float32)
    assert env.get_block_mask(state_properties, name).all()
    # Main's 1% margin admits 100.5, but the exact 101.0 boundary is excluded.
    state_properties[PROPERTY_NAMES.index("mw")] = 0.49
    assert env.get_block_mask(state_properties, name).all()
    state_properties[PROPERTY_NAMES.index("mw")] = 0.5
    assert not env.get_block_mask(state_properties, name).any()
    state_properties.fill(0)
    state_properties[PROPERTY_NAMES.index("rings")] = 1
    assert not env.get_block_mask(state_properties, name).any()
    state_properties.fill(0)
    state_properties[PROPERTY_NAMES.index("logp")] = 0.005
    assert env.get_block_mask(state_properties, name).all()
    state_properties[PROPERTY_NAMES.index("logp")] = 0.02
    assert not env.get_block_mask(state_properties, name).any()
    state_properties.fill(0)
    # Graph capacity remains strict regardless of the property margin.
    state_properties[PROPERTY_NAMES.index("heavy_atoms")] = env.max_atoms
    assert not env.get_block_mask(state_properties, name).any()

    # Separate uint8 counts must be promoted before addition, not wrap at 256.
    env.property_limits.clear()
    env.max_atoms = 255
    library.heavy_atoms[:] = 100
    state_properties.fill(0)
    state_properties[PROPERTY_NAMES.index("heavy_atoms")] = 200
    assert not env.get_block_mask(state_properties, name).any()
    env.max_atoms = 300
    assert env.get_block_mask(state_properties, name).all()


@pytest.mark.parametrize("min_library_size", [1, 2])
def test_parallel_preparation_matches_serial(
    prepared_env: Path, tmp_path: Path, min_library_size: int
) -> None:
    root = Path(__file__).parents[1]
    source = (root / "tests/fixtures/enamine_stock.smi").read_text()
    # Cross the conversion batch boundary and check that duplicate provenance
    # merges identically even when the same BB is handled by different workers.
    stock = tmp_path / "stock.smi"
    stock.write_text(source + (source.splitlines()[0] + "\n") * 512)
    parallel = tmp_path / "parallel"
    from scripts.prepare import main

    main(
        [
            "--building-blocks",
            str(stock),
            "--env-dir",
            str(parallel),
            "--template-dir",
            str(root / "data/templates"),
            "--num-workers",
            "2",
            "--min-library-size",
            str(min_library_size),
        ]
    )
    assert (parallel / "building_blocks.json").read_bytes() == (
        prepared_env / "building_blocks.json"
    ).read_bytes()
    counts = json.loads((prepared_env / "prepare_manifest.json").read_text())["stages"][
        "convert"
    ]["block_counts"]
    retained = {
        name: count for name, count in counts.items() if count >= min_library_size
    }
    excluded = {name: count for name, count in counts.items() if count < min_library_size}
    if min_library_size > 1:
        assert retained and excluded
    stage = json.loads((parallel / "prepare_manifest.json").read_text())["stages"][
        "convert"
    ]
    assert stage["min_library_size"] == min_library_size
    assert stage["block_counts"] == retained
    assert stage["excluded_block_counts"] == excluded
    assert {p.stem for p in (parallel / "blocks").glob("*.smi")} == set(retained)
    for name in retained:
        path = prepared_env / "blocks" / f"{name}.smi"
        assert (parallel / "blocks" / path.name).read_bytes() == path.read_bytes()
    with (
        np.load(parallel / "bb_feature.npz") as actual,
        np.load(prepared_env / "bb_feature.npz") as expected,
    ):
        expected_keys = [
            key
            for key in expected.files
            if key == "format" or key.split("/")[0] in retained
        ]
        assert actual.files == expected_keys
        for key in expected_keys:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_parallel_feature_failure_is_reported(tmp_path: Path) -> None:
    blocks = tmp_path / "blocks"
    blocks.mkdir()
    (blocks / "1.smi").write_text('not-a-smiles\t["id"]\n')
    with pytest.raises(ValueError):
        features_stage(tmp_path, num_workers=2)
    assert not (tmp_path / "bb_feature.npz").exists()
    assert not (tmp_path / "prepare_manifest.json").exists()


def test_loading_does_not_repeat_preparation_validation(prepared_env, monkeypatch):
    from rdkit import Chem

    from rxnflow.envs.library import load_block_libraries

    def no_parse(*args, **kwargs):
        raise AssertionError("prepared catalog must not be reparsed during load")

    original = np.lib.npyio.NpzFile.__getitem__

    def no_duplicate_smiles(self, key):
        assert not key.endswith("/smiles")
        return original(self, key)

    monkeypatch.setattr(Chem, "MolFromSmiles", no_parse)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", no_duplicate_smiles)
    assert load_block_libraries(prepared_env)


def test_batched_budgets_match_individual_masks(prepared_env):
    from rxnflow.envs.features import molecular_properties, parse_molecule

    env = SynthesisEnv(
        prepared_env, max_atoms=20, property_penalty={"mw": 200, "rings": 0, "logp": -1.0}
    )
    properties = np.stack(
        [
            molecular_properties(parse_molecule(smiles))
            for smiles in ("", "[1*]CC", "[3*]c1ccccc1")
        ]
    )
    for name, library in env.blocks.items():
        indices = np.arange(min(3, len(library)), dtype=np.int64)
        expected = np.stack(
            [env.get_block_mask(row, name, indices) for row in properties]
        )
        np.testing.assert_array_equal(
            env.get_block_mask(properties, name, indices), expected
        )


def test_conversion_filters_source_bbs_at_50_heavy_atoms(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    stock = tmp_path / "stock.smi"
    records = [
        ("C" * 49 + "Cl.[Na+]", "brick-50"),
        ("C" * 50 + "Cl", "brick-51"),
        ("Cl" + "C" * 48 + "Cl", "linker-50"),
        ("Cl" + "C" * 49 + "Cl", "linker-51"),
    ]
    stock.write_text(
        "".join(f"{smiles}\t{identifier}\n" for smiles, identifier in records)
    )
    env_dir = tmp_path / "env"
    convert_stage(stock, env_dir, root / "data/templates")
    features_stage(env_dir)
    from rxnflow.envs.library import load_block_libraries

    libraries = load_block_libraries(env_dir)
    assert any(library.is_brick for library in libraries.values())
    assert any(not library.is_brick for library in libraries.values())
    for library in libraries.values():
        assert library.heavy_atoms.dtype == np.uint8
        expected_count = 49 if library.is_brick else 48
        assert (library.heavy_atoms == expected_count).all()
        ids = {identifier for row in library.identifiers for identifier in row}
        assert ids <= {"brick-50", "linker-50"}
    # Reject 51-atom sources even when synthon conversion would remove atoms.
    # Desalting precedes the limit, so the sodium counterion does not count.
    sources = json.loads((env_dir / "building_blocks.json").read_text())
    assert set(sources) == {"brick-50", "linker-50"}
    assert all(
        Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() == 50 for smiles in sources.values()
    )
    stage = json.loads((env_dir / "prepare_manifest.json").read_text())["stages"][
        "convert"
    ]
    assert stage["max_bb_atoms"] == 50


def test_prepared_action_spaces_and_signature(prepared_env, monkeypatch):
    import hashlib

    spaces = json.loads((prepared_env / "action_space.json").read_text())
    signature = json.loads((prepared_env / "signature.json").read_text())

    def no_hash(*args, **kwargs):
        raise AssertionError("startup must use the prepared signature")

    monkeypatch.setattr(hashlib, "sha256", no_hash)
    env = SynthesisEnv(prepared_env, max_reactions=1)
    from rxnflow import __version__

    assert signature["rxnflow_version"] == __version__ == "1.0.0"
    assert env.signature == signature
    assert set(spaces) == {"initial", "reaction", "last"}
    assert [list(s.name) for s in env.initial_action_space] == spaces["initial"]
    # Derive compatibility independently from the reaction input types, including
    # both orientations. Verify preparation retained every eligible pair in order.
    for site in env.synthon_types:
        expected = [
            (name, None) for name, r in env.uni_reactions.items() if r.input_type == site
        ]
        expected += [
            (name, library)
            for name, r in env.bi_reactions.items()
            if r.state_type == site
            for library, block in env.blocks.items()
            if block.attachment_type == r.block_type
        ]
        assert [s.name for s in env.reaction_action_spaces[site]] == expected
        assert [list(pair) for pair in expected] == spaces["reaction"][str(site)]
        terminal = [
            s
            for s in env.reaction_action_spaces[site]
            if (
                env.uni_reactions[s.name[0]].output_type is None
                if s.name[1] is None
                else env.blocks[s.name[1]].is_brick
            )
        ]
        assert env.last_action_spaces[site] == terminal
        assert all(
            a is b for a, b in zip(env.last_action_spaces[site], terminal, strict=True)
        )
    expected_count = len(env.uni_reactions) + sum(
        len(env.blocks[n]) for n in env.brick_types
    )
    expected_count += sum(
        len(block)
        for r in env.bi_reactions.values()
        for block in env.blocks.values()
        if block.attachment_type == r.block_type
    )
    assert env.num_total_actions == max(2, expected_count)


def test_missing_spec_and_failed_rebuild_do_not_load_stale_spaces(
    prepared_env, tmp_path, monkeypatch
):
    import rxnflow.envs.prepare as prepare

    copied = tmp_path / "env"
    shutil.copytree(prepared_env, copied)

    def fail(*args, **kwargs):
        raise RuntimeError("feature calculation interrupted")

    monkeypatch.setattr(prepare, "block_feature_row", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        features_stage(copied)
    assert not (copied / "action_space.json").exists()
    assert not (copied / "signature.json").exists()
    with pytest.raises(FileNotFoundError, match="action_space.json"):
        SynthesisEnv(copied)
