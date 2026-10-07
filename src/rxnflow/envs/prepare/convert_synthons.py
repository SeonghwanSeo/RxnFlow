"""Convert building blocks into synthon libraries."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from functools import lru_cache, partial
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import yaml
from rdkit import Chem
from tqdm import tqdm

from rxnflow.core.synthon import (
    SynthonConversion,
    get_dummy_atoms,
    load_synthon_templates,
    typed_dummy_isotopes,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_conversions(
    templates: tuple[tuple[int, tuple[str, str]], ...],
) -> tuple[SynthonConversion, ...]:
    return tuple(
        SynthonConversion(synthon_type, *patterns) for synthon_type, patterns in templates
    )


def _convert_batch(
    records: list[tuple[str, str]],
    templates: dict[int, tuple[str, str]],
    exclude_patterns: tuple[Chem.Mol, ...] = (),
    max_atoms: int = 30,
) -> dict[str, dict[str, set[str]]]:
    """Convert a source batch into synthons with merged source IDs."""

    _smi_to_mol: dict[str, Chem.Mol] = {}

    def to_mol(smi: str) -> Chem.Mol:
        if smi not in _smi_to_mol:
            _smi_to_mol[smi] = Chem.MolFromSmiles(smi)
        return _smi_to_mol[smi]

    conversions = _get_conversions(tuple(templates.items()))

    # 1. Convert each building block into bricks.
    # One synthon can be obtained from multiple building blocks; we keep all source IDs.
    brick_dict: dict[int, dict[str, list[str]]] = {}
    for conversion in conversions:
        bricks: dict[str, list[str]] = {}
        for smiles, identifier in records:
            for brick_smi in conversion.convert(to_mol(smiles)):
                bricks.setdefault(brick_smi, []).append(identifier)
        brick_dict[conversion.synthon_type] = bricks

    # 2. Convert each unique brick to linkers. Keep both conversion orders:
    # removing one functional group can change the other template's match.
    # Sorted isotope pairs group both conversion orders into the same library.
    linker_dict: dict[tuple[int, int], dict[str, set[str]]] = {}
    for left_type, bricks in brick_dict.items():
        for brick_smi, identifiers in bricks.items():
            brick_mol = to_mol(brick_smi)
            for conversion in conversions:
                expected = tuple(sorted((left_type, conversion.synthon_type)))
                for linker_smi in conversion.convert(brick_mol):
                    linker_mol = to_mol(linker_smi)
                    # The second conversion must preserve the first handle.
                    if typed_dummy_isotopes(linker_mol) != expected:
                        continue
                    linkers = linker_dict.setdefault(expected, {})
                    linkers.setdefault(linker_smi, set()).update(identifiers)

    # 3. Filter completed synthons while retaining their chemical site types.
    def _is_valid(mol: Chem.Mol) -> bool:
        """Apply size and functional-group filters to a completed brick or linker."""
        return mol.GetNumHeavyAtoms() <= max_atoms and not any(
            mol.HasSubstructMatch(pattern) for pattern in exclude_patterns
        )

    synthons: dict[str, dict[str, set[str]]] = {}
    for synthon_type, bricks in brick_dict.items():
        for brick_smi, identifiers in bricks.items():
            mol = to_mol(brick_smi)
            if not _is_valid(mol):
                continue
            dummy_atoms = get_dummy_atoms(mol)
            assert len(dummy_atoms) == 1, f"Expected one dummy atom in {brick_smi}"
            library = synthons.setdefault(str(synthon_type), {})
            library.setdefault(brick_smi, set()).update(identifiers)

    for isotopes, linkers in linker_dict.items():
        library_name = "-".join(map(str, isotopes))
        for linker_smi, identifiers in linkers.items():
            mol = to_mol(linker_smi)
            if not _is_valid(mol):
                continue
            library = synthons.setdefault(library_name, {})
            library.setdefault(linker_smi, set()).update(identifiers)

    return synthons


def convert_synthons(
    env_dir: str | Path,
    num_workers: int = 1,
    max_atoms: int = 30,
) -> None:
    """Convert cleaned BBs into deduplicated synthon libraries."""
    started = perf_counter()
    logger.info("Preparing synthons: workers=%s, max_atoms=%s", num_workers, max_atoms)
    if max_atoms < 1:
        raise ValueError("The max_atoms value must be at least 1")
    # 1. Load cleaned building blocks, conversion rules, and exclusions.
    env_path = Path(env_dir)
    with (env_path / "building_blocks.smi").open() as f:
        records = [tuple(line.rstrip("\n").split("\t")) for line in f]
    templates = load_synthon_templates(env_path / "synthon.yaml")
    exclusion_path = env_path / "exclude_smarts.yaml"
    exclude_patterns = []
    if exclusion_path.is_file():
        with exclusion_path.open() as f:
            exclude_patterns = [Chem.MolFromSmarts(s) for s in yaml.safe_load(f)]
    logger.info(
        "Templates loaded: %s synthon types, %s exclusion patterns",
        len(templates),
        len(exclude_patterns),
    )
    synthons: dict[str, dict[str, set[str]]] = {}

    # 2. Convert batches and merge source IDs for duplicate synthons.
    # Batch conversion bounds worker input size; sorted output makes IDs deterministic.
    logger.info("Converting %d building blocks in 512-row batches", len(records))
    batches = (records[i : i + 512] for i in range(0, len(records), 512))
    convert = partial(
        _convert_batch,
        templates=templates,
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

    # 3. Assign canonical synthon IDs and save building-block provenance.
    synthon_dir = env_path / "synthons"
    synthon_dir.mkdir(exist_ok=True)
    # Remove rows left by an interrupted write before rebuilding this stage.
    for path in synthon_dir.glob("*.tsv"):
        path.unlink()
    for library_name, values in sorted(synthons.items()):
        output = synthon_dir / f"{library_name}.tsv"
        # Source IDs define the primary order; SMILES break ties when the same
        # building blocks yield several synthons of the same type.
        rows = sorted(values, key=lambda smiles: (tuple(sorted(values[smiles])), smiles))
        with output.open("w") as w:
            w.write("synthon_id\tsmiles\tbb_ids\n")
            for index, smiles in enumerate(rows):
                synthon_id = f"{library_name}_{index}"
                bb_ids = ";".join(sorted(values[smiles]))
                w.write(f"{synthon_id}\t{smiles}\t{bb_ids}\n")
        logger.info("Library %s: %d synthons -> %s", library_name, len(values), output)
    logger.info(
        "Synthon preparation complete: %d synthons, %s libraries (%.1fs)",
        sum(len(values) for values in synthons.values()),
        len(synthons),
        perf_counter() - started,
    )
