"""Assign incoming sites and connect oriented libraries to reaction actions."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rdkit import Chem

from rxnflow.core.reaction import load_reactions
from rxnflow.core.synthon import get_dummy_atoms, load_synthon_templates
from rxnflow.core.types import ActionKey

logger = logging.getLogger(__name__)


def _convert_oriented_synthons(env_path: Path, min_library_size: int) -> list[str]:
    """Write sufficiently large oriented libraries and return their names."""
    action_dir = env_path / "action_spaces"
    action_dir.mkdir(exist_ok=True)
    # Remove rows left by an interrupted write before rebuilding this stage.
    for path in action_dir.iterdir():
        path.unlink()

    # An incoming site is consumed by the action; the outgoing site retains its
    # chemical type. Equal-type asymmetric linkers can still have two directions.
    synthons: dict[str, dict[str, str]] = {}
    synthon_dir = env_path / "synthons"
    for smi_path in sorted(synthon_dir.glob("*.tsv")):
        with smi_path.open() as f:
            lines = f.readlines()[1:]  # skip header: synthon_id, smiles, bb_ids

        for line in lines:
            synthon_id, smiles, _ = line.strip().split("\t")
            mol = Chem.MolFromSmiles(smiles)
            sites = get_dummy_atoms(mol)
            assert len(sites) in (1, 2)
            if len(sites) == 1:
                library_name = str(sites[0].GetIsotope())
                library = synthons.setdefault(library_name, {})
                sites[0].SetIsotope(0)
                oriented_smi = Chem.MolToSmiles(mol)
                library[oriented_smi] = synthon_id
            else:
                assert len(sites) == 2
                left, right = sites
                for incoming, outgoing in ((left, right), (right, left)):
                    _incoming_type = str(incoming.GetIsotope())
                    _outgoing_type = str(outgoing.GetIsotope())
                    library_name = "-".join((_incoming_type, _outgoing_type))
                    library = synthons.setdefault(library_name, {})
                    _mol = Chem.Mol(mol)  # copy to avoid mutating the original
                    _mol.GetAtomWithIdx(incoming.GetIdx()).SetIsotope(0)
                    oriented_smiles = Chem.MolToSmiles(_mol)
                    library[oriented_smiles] = synthon_id

    library_names = []
    for name, rows in sorted(synthons.items()):
        if len(rows) < min_library_size:
            continue
        with (action_dir / f"{name}.smi").open("w") as w:
            for smiles, synthon_id in sorted(rows.items()):
                w.write(f"{smiles}\t{synthon_id}\n")
        library_names.append(name)
    if not library_names:
        raise ValueError(f"No synthon libraries meet min_library_size={min_library_size}")
    return library_names


def _write_action_space(env_path: Path, libraries: list[str]) -> None:
    """Connect retained libraries to compatible reactions and save the action space."""
    synthon_types = set(load_synthon_templates(env_path / "synthon.yaml"))
    uni, bi = load_reactions(env_path / "reaction.yaml")
    bricks = []
    by_attachment: dict[int, list[str]] = {}
    for name in libraries:
        sites = tuple(int(value) for value in name.split("-"))
        by_attachment.setdefault(sites[0], []).append(name)
        if len(sites) == 1:
            bricks.append(name)
    if not bricks:
        raise ValueError("Prepared environment contains no one-site bricks")
    reactions: dict[int, list[ActionKey]] = {site: [] for site in sorted(synthon_types)}
    for name, reaction in uni.items():
        reactions[reaction.input_type].append((name, None))
    for name, reaction in bi.items():
        for library in by_attachment.get(reaction.attachment_type, []):
            reactions[reaction.state_type].append((name, library))
    spaces = {
        "initial": [("first_synthon", name) for name in bricks],
        "reaction": reactions,
    }
    action_path = env_path / "action_space.json"
    with action_path.open("w") as w:
        json.dump(spaces, w, indent=2)


def build_action_space(env_dir: str | Path, min_library_size: int = 10) -> list[str]:
    """Orient synthons, filter library sizes, and write action eligibility."""
    env_path = Path(env_dir)
    library_names = _convert_oriented_synthons(env_path, min_library_size)
    _write_action_space(env_path, library_names)
    return library_names
