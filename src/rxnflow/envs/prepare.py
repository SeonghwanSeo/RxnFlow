"""Building-block synthon conversion and NumPy feature preparation."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import lru_cache, partial
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import cast
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from tqdm import tqdm

from rxnflow import __version__
from rxnflow.core.reaction import load_reactions
from rxnflow.core.synthon import (
    SynthonConversion,
    SynthonSpec,
    load_synthon_specs,
    typed_dummy_isotopes,
)
from rxnflow.envs.features import (
    FINGERPRINT_DIM,
    PROPERTY_DIM,
    synthon_feature_row,
)

logger = logging.getLogger(__name__)

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


def _read_building_blocks(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    salt_remover = SaltRemover()
    logger.info(f"Reading and cleaning building blocks: {path}")
    line_number = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(
            tqdm(
                handle,
                desc="Cleaning building blocks",
                unit="row",
            ),
            start=1,
        ):
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) != 2 or not fields[0] or not fields[1]:
                raise ValueError(f"{path}:{line_number}: expected SMILES<TAB>identifier")
            smiles = _clean_smiles(fields[0], salt_remover)
            if smiles is not None:
                records.append((smiles, fields[1].strip()))
    logger.info(
        f"Building blocks: read={line_number:,}, retained={len(records):,}, rejected={line_number - len(records):,}"
    )
    if not records:
        raise ValueError(f"no valid building blocks in {path}")
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
    records: list[tuple[str, str]],
    specs: list[SynthonSpec],
    exclude_patterns: tuple[Chem.Mol, ...] = (),
    max_atoms: int = 30,
) -> dict[str, dict[str, set[str]]]:
    """Convert a source batch into oriented synthons with merged source IDs."""
    # 1. Reuse compiled conversions and collect one-handle products.
    conversions = _conversion_templates(tuple(specs))
    synthons: dict[str, dict[str, set[str]]] = {}
    bricks: dict[int, dict[str, tuple[Chem.Mol, set[str]]]] = {
        spec.type: {} for spec in specs
    }

    # Parse each source once for all conversion templates.
    for smiles, identifier in records:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        for conversion in conversions:
            synthon_type = conversion.spec.type
            for product, product_mol in conversion.run_mol(mol).items():
                if typed_dummy_isotopes(product_mol) != (synthon_type,):
                    continue
                bricks[synthon_type].setdefault(product, (product_mol, set()))[1].add(
                    identifier
                )

    # 2. Convert bricks to two-handle linkers and store both attachment directions.
    # Try both conversion orders: removing one group can change the other match.
    for left_type, values in bricks.items():
        for mol, identifiers in values.values():
            for right in conversions:
                expected = tuple(sorted((left_type, right.spec.type)))
                for product_mol in right.run_mol(mol).values():
                    if typed_dummy_isotopes(product_mol) != expected:
                        continue
                    # Filter completed linkers, never the intermediate bricks:
                    # the second conversion may consume an excluded group.
                    # RDKit heavy atoms exclude hydrogen and dummy handles.
                    if product_mol.GetNumHeavyAtoms() > max_atoms:
                        continue
                    if any(product_mol.HasSubstructMatch(q) for q in exclude_patterns):
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
                        library_name = (
                            f"{attachment.GetIsotope()}-"
                            f"{oriented.GetAtomWithIdx(other).GetIsotope()}"
                        )
                        attachment.SetIsotope(0)
                        smiles = Chem.MolToSmiles(oriented)
                        synthons.setdefault(library_name, {}).setdefault(
                            smiles, set()
                        ).update(identifiers)

    # 3. Store bricks with the same isotope-0 attachment marker as linkers.
    # FirstSynthon restores its chemical type from the library key.
    for synthon_type, values in bricks.items():
        key = str(synthon_type)
        for original, identifiers in values.values():
            mol = Chem.Mol(original)
            if mol.GetNumHeavyAtoms() > max_atoms:
                continue
            if any(mol.HasSubstructMatch(q) for q in exclude_patterns):
                continue
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetIsotope(0)
            smiles = Chem.MolToSmiles(mol)
            synthons.setdefault(key, {}).setdefault(smiles, set()).update(identifiers)

    return synthons


def convert_stage(
    building_blocks: str | Path,
    env_dir: str | Path,
    config_path: str | Path,
    num_workers: int = 1,
    min_library_size: int = 10,
    max_atoms: int = 30,
    druglikeness_threshold: float | None = None,
    druglikeness_device: str = "cpu",
) -> None:
    """Write synthon libraries and source provenance from a building-block catalog."""
    started = perf_counter()
    logger.info(
        f"Preparing synthons: workers={num_workers}, min_library_size={min_library_size}, max_atoms={max_atoms}"
    )
    # 1. Read templates and clean source BBs, before synthon conversion.
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if min_library_size < 1:
        raise ValueError("min_library_size must be at least 1")
    if max_atoms < 1:
        raise ValueError("max_atoms must be at least 1")
    env_path = Path(env_dir)
    source_path = Path(building_blocks)
    config_path = Path(config_path)
    template_path = config_path.parent
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with config_path.open(encoding="utf-8") as handle:
        template_config = yaml.safe_load(handle)
    synthon_path = template_path / template_config["synthon"]
    reaction_path = template_path / template_config["reaction"]
    exclusion_path = (
        template_path / template_config["exclude_smarts"]
        if template_config.get("exclude_smarts")
        else None
    )
    exclude_smarts: list[str] = []
    if exclusion_path is not None:
        with exclusion_path.open(encoding="utf-8") as handle:
            exclude_smarts = yaml.safe_load(handle)
        if not isinstance(exclude_smarts, list):
            raise ValueError("exclude_smarts must contain a YAML list of SMARTS strings")
    exclude_patterns = []
    for smarts in exclude_smarts:
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise ValueError(f"invalid exclusion SMARTS: {smarts}")
        exclude_patterns.append(query)
    if not reaction_path.is_file():
        raise FileNotFoundError(reaction_path)
    specs = load_synthon_specs(synthon_path)
    logger.info(
        f"Templates loaded: {len(specs)} synthon types, {len(exclude_patterns)} exclusion patterns"
    )
    records = _read_building_blocks(source_path)
    # Screen cleaned source building blocks before creating their oriented synthons.
    num_input = len(records)
    if druglikeness_threshold is not None:
        from druglikeness.deepdl import DeepDL

        if not 0 <= druglikeness_threshold <= 100:
            raise ValueError("druglikeness threshold must be between 0 and 100")
        logger.info(
            f"Loading DeepDL extended: device={druglikeness_device}, threshold={druglikeness_threshold}"
        )
        model = DeepDL.from_pretrained("extended", device=druglikeness_device)
        batch_size = 256 if druglikeness_device.startswith("cuda") else 64
        retained = []
        for start in range(0, len(records), 100_000):
            chunk = records[start : start + 100_000]
            scores = model.screening(
                [smiles for smiles, _ in chunk],
                naive=True,
                batch_size=batch_size,
                verbose=True,
            )
            retained.extend(
                record
                for record, score in zip(chunk, scores, strict=True)
                if score >= druglikeness_threshold
            )
        records = retained
        logger.info(f"Druglikeness: retained {len(records):,} / {num_input:,} building blocks")
        if not records:
            raise ValueError("no building blocks passed druglikeness filtering")
        del model

    synthons: dict[str, dict[str, set[str]]] = {}
    sources: dict[str, str] = {}

    for smiles, identifier in records:
        if identifier in sources and sources[identifier] != smiles:
            raise ValueError(f"building-block ID has multiple structures: {identifier}")
        sources[identifier] = smiles

    # 2. Convert source batches and merge duplicate synthons/source IDs.
    # Small batches distribute expensive conversions without sending the entire
    # catalog to every process. Ordered merging and sorted output keep row IDs
    # identical for serial and parallel preparation.
    logger.info(f"Converting {len(records):,} building blocks in 512-row batches")
    batches = (records[i : i + 512] for i in range(0, len(records), 512))
    convert = partial(
        _convert_batch,
        specs=specs,
        exclude_patterns=tuple(exclude_patterns),
        max_atoms=max_atoms,
    )
    context = get_context("spawn").Pool(num_workers) if num_workers > 1 else nullcontext()
    with context as pool:
        results = (
            pool.imap(convert, batches) if pool is not None else map(convert, batches)
        )
        for result in tqdm(
            results,
            total=(len(records) + 511) // 512,
            desc="Converting synthons",
            unit="batch",
        ):
            for library_name, values in result.items():
                target = synthons.setdefault(library_name, {})
                for smiles, identifiers in values.items():
                    target.setdefault(smiles, set()).update(identifiers)

    # 3. Apply minimum library size to globally deduplicated oriented synthons.
    # Count unique oriented synthons after merging all source batches. Supplier
    # IDs and repeated source rows do not increase a library's size.
    excluded_counts = {
        name: len(values)
        for name, values in synthons.items()
        if len(values) < min_library_size
    }
    if len(excluded_counts) == len(synthons):
        raise ValueError("no synthon libraries meet min_library_size")

    logger.info(
        f"Conversion produced {sum(len(v) for v in synthons.values()):,} unique oriented synthons in {len(synthons)} libraries; excluded {len(excluded_counts)} libraries below min_library_size"
    )
    logger.info(f"Writing synthon libraries and provenance: {env_path}")
    # 4. Write sorted library rows, source provenance, and the active templates.
    env_path.mkdir(parents=True, exist_ok=True)
    (env_path / "signature.json").unlink(missing_ok=True)
    (env_path / "action_space.json").unlink(missing_ok=True)
    synthon_dir = env_path / "synthons"
    synthon_dir.mkdir(exist_ok=True)
    for old in synthon_dir.glob("*.smi"):
        old.unlink()
    counts: dict[str, int] = {}
    for library_name, values in sorted(synthons.items()):
        if len(values) < min_library_size:
            continue
        output = synthon_dir / f"{library_name}.smi"
        temporary = output.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for smiles in sorted(values):
                identifiers = json.dumps(sorted(values[smiles]))
                handle.write(f"{smiles}\t{identifiers}\n")
        temporary.replace(output)
        counts[library_name] = len(values)
        logger.info(f"Library {library_name}: {len(values):,} synthons -> {output}")
    # One source record can map to many synthons, and identical synthons may
    # have several suppliers' IDs. Keep this provenance outside the MDP state.
    (env_path / "building_blocks.json").write_text(
        json.dumps(sources, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(synthon_path, env_path / "synthon.yaml")
    shutil.copyfile(reaction_path, env_path / "reaction.yaml")
    # Snapshot resolved inputs so the prepared environment is self-contained.
    resolved_config = {"reaction": "reaction.yaml", "synthon": "synthon.yaml"}
    if exclusion_path is not None:
        shutil.copyfile(exclusion_path, env_path / "exclude_smarts.yaml")
        resolved_config["exclude_smarts"] = "exclude_smarts.yaml"
    else:
        (env_path / "exclude_smarts.yaml").unlink(missing_ok=True)
    (env_path / "config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )

    # 5. Invalidate features because the library row indices may have changed.
    (env_path / "synthon_features.npz").unlink(missing_ok=True)
    manifest = {"format": MANIFEST_FORMAT, "stages": {}}
    (env_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    _complete_stage(
        env_path,
        "convert",
        {
            "synthon_counts": counts,
            "max_synthon_atoms": max_atoms,
            "exclude_smarts": exclude_smarts,
            "druglikeness": {
                "threshold": druglikeness_threshold,
                "model": "extended" if druglikeness_threshold is not None else None,
                "input_count": num_input,
                "retained_count": len(records),
            },
            "min_library_size": min_library_size,
            "excluded_synthon_counts": excluded_counts,
        },
    )

    logger.info(
        f"Synthon preparation complete: {sum(counts.values()):,} synthons, {len(counts)} libraries ({perf_counter() - started:.1f}s)"
    )


def features_stage(env_dir: str | Path, num_workers: int = 1) -> None:
    """Write properties, fingerprints, and atom counts in each library's row order."""
    started = perf_counter()
    logger.info(f"Preparing features: env={env_dir}, workers={num_workers}")
    # 1. Enumerate converted libraries before opening the replacement archive.
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    env_path = Path(env_dir)
    synthon_dir = env_path / "synthons"
    if not synthon_dir.is_dir():
        raise FileNotFoundError(synthon_dir)
    files = sorted(synthon_dir.glob("*.smi"))
    if not files:
        raise ValueError(f"no .smi files in {synthon_dir}")
    # The signature and action space must never outlive an interrupted feature rebuild.
    (env_path / "signature.json").unlink(missing_ok=True)
    (env_path / "action_space.json").unlink(missing_ok=True)
    counts: dict[str, int] = {}
    output = env_path / "synthon_features.npz"
    temporary = output.with_suffix(".tmp")
    context = get_context("spawn").Pool(num_workers) if num_workers > 1 else nullcontext()
    # 2. Calculate rows and stream one library at a time into the NPZ archive.
    # NPZ is a zip of NPY arrays. Stream one library at a time instead of
    # retaining the entire catalog's arrays and millions of temporary row arrays.
    with context as pool, ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        with archive.open("format.npy", "w") as entry:
            np.lib.format.write_array(
                entry, np.array("rxnflow-synthon-feature"), allow_pickle=False
            )
        for path in files:
            smiles = _read_prepared_smiles(path)
            count = len(smiles)
            counts[path.stem] = count
            if not count:
                raise ValueError(f"empty synthon file: {path}")
            logger.info(f"Features for library {path.stem}: {count:,} synthons")
            properties = np.empty((count, PROPERTY_DIM), dtype=np.float32)
            fingerprints = np.empty((count, FINGERPRINT_DIM), dtype=np.uint8)
            heavy_atoms = np.empty(count, dtype=np.uint8)
            rows = (
                pool.imap(synthon_feature_row, smiles, chunksize=256)
                if pool is not None
                else map(synthon_feature_row, smiles)
            )
            for index, (prop, fingerprint, atom_count) in enumerate(
                tqdm(
                    rows,
                    total=count,
                    desc=f"Features {path.stem}",
                    unit="synthon",
                )
            ):
                properties[index] = prop
                fingerprints[index] = fingerprint
                heavy_atoms[index] = atom_count
            logger.info(f"Writing features for library {path.stem}")
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
    # 3. Publish the archive only after every library has been written.
    temporary.replace(output)
    # 4. Finalize static eligibility and identity before marking the stage complete.
    logger.info("Building action spaces and computing environment signature")
    action_names = _write_action_space(env_path, counts)
    _write_signature(env_path, counts, action_names)
    _complete_stage(
        env_path,
        "features",
        {"library_names": sorted(path.stem for path in synthon_dir.glob("*.smi"))},
    )

    logger.info(
        f"Feature preparation complete: {sum(counts.values()):,} synthons, {len(action_names)} actions -> {output} ({perf_counter() - started:.1f}s)"
    )


def _write_action_space(env_path: Path, counts: dict[str, int]) -> list[str]:
    """Write chemical eligibility; runtime applies synthesis budgets."""
    # Keep SMARTS and library descriptors in their existing source files.
    # The spec stores only pairs whose compatibility would otherwise be rebuilt.
    synthon_types = {spec.type for spec in load_synthon_specs(env_path / "synthon.yaml")}
    uni, bi = load_reactions(env_path / "reaction.yaml")
    bricks = []
    by_attachment: dict[int, list[str]] = {}
    # Counts follow sorted .smi filenames, matching the library loader order.
    for name in counts:
        sites = tuple(int(value) for value in name.split("-"))
        if len(sites) not in (1, 2) or not set(sites) <= synthon_types:
            raise ValueError(f"invalid brick/linker type: {name}")
        by_attachment.setdefault(sites[0], []).append(name)
        if len(sites) == 1:
            bricks.append(name)
    if not bricks:
        raise ValueError("prepared environment contains no one-site bricks")
    reactions: dict[int, list[tuple[str, str | None]]] = {
        site: [] for site in sorted(synthon_types)
    }
    for name, reaction in uni.items():
        if reaction.input_type not in synthon_types or (
            reaction.output_type is not None and reaction.output_type not in synthon_types
        ):
            raise ValueError(f"unknown synthon type in {name}")
        reactions[reaction.input_type].append((name, None))
    for name, reaction in bi.items():
        if not set(reaction.synthon_types) <= synthon_types:
            raise ValueError(f"unknown synthon type in {name}")
        for library in by_attachment.get(reaction.attachment_type, []):
            reactions[reaction.state_type].append((name, library))
    spaces = {
        "initial": [("first_synthon", name) for name in bricks],
        "reaction": reactions,
    }
    action_names = ["first_synthon", *sorted(uni), *sorted(bi)]
    if len(set(action_names)) != len(action_names):
        raise ValueError("reaction action names must be unique")

    # Publish the spec only after every compatible pair is assembled.
    action_path = env_path / "action_space.json"
    temporary = action_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(spaces, indent=2) + "\n", encoding="utf-8")
    temporary.replace(action_path)
    return action_names


def _write_signature(
    env_path: Path, counts: dict[str, int], action_names: list[str]
) -> None:
    """Publish identity after templates, rows, features and eligibility are complete."""
    digest = hashlib.sha256()
    for name in (
        "config.yaml",
        "action_space.json",
        "synthon.yaml",
        "reaction.yaml",
        "building_blocks.json",
        "synthon_features.npz",
    ):
        with (env_path / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    if (env_path / "exclude_smarts.yaml").exists():
        digest.update((env_path / "exclude_smarts.yaml").read_bytes())
    for name in sorted(counts):
        digest.update(name.encode())
        with (env_path / "synthons" / f"{name}.smi").open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    # Publish identity last: it describes all completed data, including action space.
    signature = {
        "format": "rxnflow-env",
        "rxnflow_version": __version__,
        "synthon_counts": counts,
        "reaction_names": action_names,
        "content_sha256": digest.hexdigest(),
    }
    path = env_path / "signature.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
