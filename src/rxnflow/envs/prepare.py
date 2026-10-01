"""Enamine synthon conversion and NumPy feature preparation."""

from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

from rxnflow.envs.chemistry.features import block_feature_row
from rxnflow.envs.chemistry.synthon import (
    SynthonConversion,
    load_synthon_specs,
    typed_dummy_isotopes,
)

ALLOWED_ATOMIC_NUMBERS = {5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 85}
MANIFEST_NAME = "prepare_manifest.json"
MANIFEST_FORMAT = "rxnflow-prepare"


def _load_manifest(env_dir: Path) -> dict[str, object]:
    path = env_dir / MANIFEST_NAME
    if not path.exists():
        return {"format": MANIFEST_FORMAT, "stages": {}}
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != MANIFEST_FORMAT or not isinstance(
        manifest.get("stages"), dict
    ):
        raise ValueError(f"unsupported preparation manifest: {path}")
    return manifest


def _complete_stage(env_dir: Path, stage: str, details: dict[str, object]) -> None:
    manifest = _load_manifest(env_dir)
    stages = cast(dict[str, object], manifest["stages"])
    stages[stage] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    path = env_dir / MANIFEST_NAME
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _clean_smiles(smiles: str, salt_remover: SaltRemover) -> str | None:
    if "[2H]" in smiles or "[13C" in smiles or "[13c" in smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Small catalog reagents (e.g. acetic acid) can themselves match a salt
    # pattern. Desalting must not discard the entire building block.
    stripped = salt_remover.StripMol(mol, dontRemoveEverything=True)
    if stripped is None or not stripped.GetNumAtoms():
        return None
    if {atom.GetAtomicNum() for atom in stripped.GetAtoms()} - ALLOWED_ATOMIC_NUMBERS:
        return None
    canonical = Chem.MolToSmiles(stripped)
    return None if "." in canonical else canonical


def _read_enamine(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    salt_remover = SaltRemover()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) != 2 or not fields[0] or not fields[1]:
                raise ValueError(f"{path}:{line_number}: expected SMILES<TAB>identifier")
            smiles = _clean_smiles(fields[0], salt_remover)
            if smiles is not None:
                records.append((smiles, fields[1].strip()))
    if not records:
        raise ValueError(f"no valid Enamine building blocks in {path}")
    return records


def _read_prepared_smiles(path: Path) -> list[str]:
    smiles: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) != 2 or not fields[0] or not fields[1]:
                raise ValueError(
                    f"{path}:{line_number}: expected synthon<TAB>identifiers"
                )
            smiles.append(fields[0])
    return smiles


def _add_product(
    output: dict[str, OrderedDict[str, set[str]]],
    block_type: str,
    smiles: str,
    identifiers: set[str],
) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    output.setdefault(block_type, OrderedDict()).setdefault(smiles, set()).update(
        identifiers
    )


def convert_stage(
    building_blocks: str | Path,
    env_dir: str | Path,
    template_dir: str | Path,
) -> None:
    env_path = Path(env_dir)
    source_path = Path(building_blocks)
    template_path = Path(template_dir)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    for name in ("synthon.yaml", "reaction.yaml"):
        if not (template_path / name).is_file():
            raise FileNotFoundError(template_path / name)

    specs = load_synthon_specs(template_path / "synthon.yaml")
    conversions = [SynthonConversion(spec) for spec in specs]
    records = _read_enamine(source_path)
    blocks: dict[str, OrderedDict[str, set[str]]] = {}

    bricks: dict[int, OrderedDict[str, set[str]]] = {}
    for conversion in conversions:
        block_type = conversion.spec.type
        converted: OrderedDict[str, set[str]] = OrderedDict()
        for smiles, identifier in records:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            for product in conversion.run_mol(mol):
                product_mol = Chem.MolFromSmiles(product)
                if product_mol is None or typed_dummy_isotopes(product_mol) != (
                    block_type,
                ):
                    continue
                converted.setdefault(product, set()).add(identifier)
                _add_product(blocks, str(block_type), product, {identifier})
        bricks[block_type] = converted

    # Try both orders: removing one group can change the SMARTS context of the
    # other. Deduplicate by the final synthon, including its typed handles.
    for left in conversions:
        left_type = left.spec.type
        for right in conversions:
            right_type = right.spec.type
            block_type = "-".join(map(str, sorted((left_type, right_type))))
            for brick, identifiers in bricks[left_type].items():
                mol = Chem.MolFromSmiles(brick)
                assert mol is not None
                for product in right.run_mol(mol):
                    product_mol = Chem.MolFromSmiles(product)
                    expected = tuple(sorted((left_type, right_type)))
                    if (
                        product_mol is None
                        or typed_dummy_isotopes(product_mol) != expected
                    ):
                        continue
                    _add_product(blocks, block_type, product, identifiers)

    env_path.mkdir(parents=True, exist_ok=True)
    block_dir = env_path / "blocks"
    block_dir.mkdir(exist_ok=True)
    for old in block_dir.glob("*.smi"):
        old.unlink()
    counts: dict[str, int] = {}
    for block_type, values in sorted(blocks.items()):
        if not values:
            continue
        output = block_dir / f"{block_type}.smi"
        temporary = output.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for smiles in sorted(values):
                identifiers = json.dumps(sorted(values[smiles]))
                handle.write(f"{smiles}\t{identifiers}\n")
        temporary.replace(output)
        counts[block_type] = len(values)
    if not counts:
        raise ValueError("synthon conversion produced no block libraries")

    # One source record can map to many synthons, and identical synthons may
    # have several suppliers' IDs. Keep this provenance outside the MDP state.
    sources: dict[str, str] = {}
    for smiles, identifier in records:
        if identifier in sources and sources[identifier] != smiles:
            raise ValueError(f"building-block ID has multiple structures: {identifier}")
        sources[identifier] = smiles
    (env_path / "building_blocks.json").write_text(
        json.dumps(sources, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(template_path / "synthon.yaml", env_path / "synthon.yaml")
    shutil.copyfile(template_path / "reaction.yaml", env_path / "reaction.yaml")

    # Re-conversion invalidates the previous aligned features and stage record.
    (env_path / "bb_feature.npz").unlink(missing_ok=True)
    manifest = {"format": MANIFEST_FORMAT, "stages": {}}
    (env_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    _complete_stage(env_path, "convert", {"block_counts": counts})


def features_stage(env_dir: str | Path) -> None:
    env_path = Path(env_dir)
    block_dir = env_path / "blocks"
    if not block_dir.is_dir():
        raise FileNotFoundError(block_dir)
    arrays: dict[str, np.ndarray] = {"format": np.array("rxnflow-bb-feature")}
    for path in sorted(block_dir.glob("*.smi")):
        smiles = _read_prepared_smiles(path)
        if not smiles:
            raise ValueError(f"empty building-block file: {path}")
        properties = []
        fingerprints = []
        heavy_atoms = []
        for value in smiles:
            prop, fingerprint, atom_count = block_feature_row(value)
            properties.append(prop)
            fingerprints.append(fingerprint)
            heavy_atoms.append(atom_count)
        # Flat array keys keep NPZ directly readable with NumPy; no nested
        # Python objects are serialized. SMILES preserve exact row alignment.
        arrays[f"{path.stem}/smiles"] = np.array(smiles)
        arrays[f"{path.stem}/properties"] = np.stack(properties)
        arrays[f"{path.stem}/fingerprints"] = np.stack(fingerprints)
        arrays[f"{path.stem}/heavy_atoms"] = np.array(heavy_atoms, dtype=np.int32)
    if len(arrays) == 1:
        raise ValueError(f"no .smi files in {block_dir}")
    output = env_path / "bb_feature.npz"
    np.savez_compressed(output, **arrays)
    _complete_stage(
        env_path,
        "features",
        {"block_types": sorted(path.stem for path in block_dir.glob("*.smi"))},
    )
