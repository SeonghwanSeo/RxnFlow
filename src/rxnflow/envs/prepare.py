"""Enamine synthon conversion and NumPy feature preparation."""

from __future__ import annotations

import json
import shutil
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import lru_cache, partial
from multiprocessing import get_context
from pathlib import Path
from typing import cast
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

from rxnflow.envs.chemistry.features import (
    FINGERPRINT_DIM,
    PROPERTY_DIM,
    block_feature_row,
)
from rxnflow.envs.chemistry.synthon import (
    SynthonConversion,
    SynthonSpec,
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


@lru_cache(maxsize=1)
def _conversion_templates(
    specs: tuple[SynthonSpec, ...],
) -> tuple[SynthonConversion, ...]:
    # Each preparation process compiles the current template set once, not once
    # per 512-row task. Reactions are read-only during RunReactants.
    return tuple(SynthonConversion(spec) for spec in specs)


def _convert_batch(
    records: list[tuple[str, str]], specs: list[SynthonSpec]
) -> dict[str, dict[str, set[str]]]:
    # Each worker reuses its RDKit reactions and deduplicates a bounded source batch.
    # The parent merges source-ID sets across batches before writing sorted rows.
    conversions = _conversion_templates(tuple(specs))
    blocks: dict[str, dict[str, set[str]]] = {}
    bricks: dict[int, dict[str, tuple[Chem.Mol, set[str]]]] = {
        spec.type: {} for spec in specs
    }

    # Parse each source once, then try all conversions on that same read-only
    # molecule. Reversing these loops reparses every BB 35 times.
    for smiles, identifier in records:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        for conversion in conversions:
            block_type = conversion.spec.type
            for product, product_mol in conversion.run_mol(mol).items():
                if typed_dummy_isotopes(product_mol) != (block_type,):
                    continue
                bricks[block_type].setdefault(product, (product_mol, set()))[1].add(
                    identifier
                )

    # Try both orders: removing one group can change the SMARTS context of the
    # other. Keep each brick Mol for all second conversions as well.
    for left_type, values in bricks.items():
        for mol, identifiers in values.values():
            for right in conversions:
                expected = tuple(sorted((left_type, right.spec.type)))
                for product_mol in right.run_mol(mol).values():
                    if typed_dummy_isotopes(product_mol) != expected:
                        continue
                    # Store each attachment direction as a separate library row.
                    # Isotope 0 marks the incoming attachment; the other handle
                    # keeps its type and becomes the next state's open site.
                    sites = [
                        atom.GetIdx()
                        for atom in product_mol.GetAtoms()
                        if atom.GetAtomicNum() == 0
                    ]
                    for site, other in (sites, sites[::-1]):
                        oriented = Chem.Mol(product_mol)
                        attachment = oriented.GetAtomWithIdx(site)
                        block_type = (
                            f"{attachment.GetIsotope()}-"
                            f"{oriented.GetAtomWithIdx(other).GetIsotope()}"
                        )
                        attachment.SetIsotope(0)
                        smiles = Chem.MolToSmiles(oriented)
                        blocks.setdefault(block_type, {}).setdefault(
                            smiles, set()
                        ).update(identifiers)

    # Bricks use the same attachment convention as linkers. FirstBlock restores
    # their type from the library key when creating the initial synthon state.
    for block_type, values in bricks.items():
        key = str(block_type)
        for original, identifiers in values.values():
            mol = Chem.Mol(original)
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetIsotope(0)
            smiles = Chem.MolToSmiles(mol)
            blocks.setdefault(key, {}).setdefault(smiles, set()).update(identifiers)

    return blocks


def convert_stage(
    building_blocks: str | Path,
    env_dir: str | Path,
    template_dir: str | Path,
    num_workers: int = 1,
    min_library_size: int = 1,
) -> None:
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if min_library_size < 1:
        raise ValueError("min_library_size must be at least 1")
    env_path = Path(env_dir)
    source_path = Path(building_blocks)
    template_path = Path(template_dir)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    for name in ("synthon.yaml", "reaction.yaml"):
        if not (template_path / name).is_file():
            raise FileNotFoundError(template_path / name)

    specs = load_synthon_specs(template_path / "synthon.yaml")
    records = _read_enamine(source_path)
    blocks: dict[str, dict[str, set[str]]] = {}
    sources: dict[str, str] = {}

    for smiles, identifier in records:
        if identifier in sources and sources[identifier] != smiles:
            raise ValueError(f"building-block ID has multiple structures: {identifier}")
        sources[identifier] = smiles

    # Small batches distribute expensive conversions without sending the entire
    # catalog to every process. Ordered merging and sorted output keep row IDs
    # identical for serial and parallel preparation.
    batches = (records[i : i + 512] for i in range(0, len(records), 512))
    convert = partial(_convert_batch, specs=specs)
    context = get_context("spawn").Pool(num_workers) if num_workers > 1 else nullcontext()
    with context as pool:
        results = (
            pool.imap(convert, batches) if pool is not None else map(convert, batches)
        )
        for result in results:
            for block_type, values in result.items():
                target = blocks.setdefault(block_type, {})
                for smiles, identifiers in values.items():
                    target.setdefault(smiles, set()).update(identifiers)

    # Count unique oriented synthons after merging all source batches. Supplier
    # IDs and repeated source rows do not increase a library's size.
    excluded_counts = {
        name: len(values)
        for name, values in blocks.items()
        if len(values) < min_library_size
    }
    if len(excluded_counts) == len(blocks):
        raise ValueError("no block libraries meet min_library_size")

    env_path.mkdir(parents=True, exist_ok=True)
    block_dir = env_path / "blocks"
    block_dir.mkdir(exist_ok=True)
    for old in block_dir.glob("*.smi"):
        old.unlink()
    counts: dict[str, int] = {}
    for block_type, values in sorted(blocks.items()):
        if len(values) < min_library_size:
            continue
        output = block_dir / f"{block_type}.smi"
        temporary = output.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for smiles in sorted(values):
                identifiers = json.dumps(sorted(values[smiles]))
                handle.write(f"{smiles}\t{identifiers}\n")
        temporary.replace(output)
        counts[block_type] = len(values)
    # One source record can map to many synthons, and identical synthons may
    # have several suppliers' IDs. Keep this provenance outside the MDP state.
    (env_path / "building_blocks.json").write_text(
        json.dumps(sources, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(template_path / "synthon.yaml", env_path / "synthon.yaml")
    shutil.copyfile(template_path / "reaction.yaml", env_path / "reaction.yaml")

    # Re-conversion invalidates the previous aligned features and stage record.
    (env_path / "bb_feature.npz").unlink(missing_ok=True)
    manifest = {"format": MANIFEST_FORMAT, "stages": {}}
    (env_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    _complete_stage(
        env_path,
        "convert",
        {
            "block_counts": counts,
            "min_library_size": min_library_size,
            "excluded_block_counts": excluded_counts,
        },
    )


def features_stage(env_dir: str | Path, num_workers: int = 1) -> None:
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    env_path = Path(env_dir)
    block_dir = env_path / "blocks"
    if not block_dir.is_dir():
        raise FileNotFoundError(block_dir)
    files = sorted(block_dir.glob("*.smi"))
    if not files:
        raise ValueError(f"no .smi files in {block_dir}")
    output = env_path / "bb_feature.npz"
    temporary = output.with_suffix(".tmp")
    context = get_context("spawn").Pool(num_workers) if num_workers > 1 else nullcontext()
    # NPZ is a zip of NPY arrays. Stream one library at a time instead of
    # retaining the entire catalog's arrays and millions of temporary row arrays.
    with context as pool, ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        with archive.open("format.npy", "w") as entry:
            np.lib.format.write_array(
                entry, np.array("rxnflow-bb-feature"), allow_pickle=False
            )
        for path in files:
            smiles = _read_prepared_smiles(path)
            count = len(smiles)
            if not count:
                raise ValueError(f"empty building-block file: {path}")
            properties = np.empty((count, PROPERTY_DIM), dtype=np.float32)
            fingerprints = np.empty((count, FINGERPRINT_DIM), dtype=np.uint8)
            heavy_atoms = np.empty(count, dtype=np.int32)
            rows = (
                pool.imap(block_feature_row, smiles, chunksize=256)
                if pool is not None
                else map(block_feature_row, smiles)
            )
            for index, (prop, fingerprint, atom_count) in enumerate(rows):
                properties[index] = prop
                fingerprints[index] = fingerprint
                heavy_atoms[index] = atom_count
            for name, array in (
                ("smiles", np.array(smiles)),
                ("properties", properties),
                ("fingerprints", fingerprints),
                ("heavy_atoms", heavy_atoms),
            ):
                with archive.open(
                    f"{path.stem}/{name}.npy", "w", force_zip64=True
                ) as entry:
                    np.lib.format.write_array(entry, array, allow_pickle=False)
    temporary.replace(output)
    _complete_stage(
        env_path,
        "features",
        {"block_types": sorted(path.stem for path in block_dir.glob("*.smi"))},
    )
