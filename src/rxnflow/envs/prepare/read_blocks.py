"""Read, canonicalize, and optionally screen building blocks."""

from __future__ import annotations

import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from tqdm import tqdm

logger = logging.getLogger(__name__)

ALLOWED_ATOMIC_TYPES = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}


def _read_building_blocks(path: Path) -> list[tuple[str, str]]:
    """Read a tab-delimited file of SMILES and building-block identifiers."""
    logger.info("Reading building blocks: %s", path)

    records: list[tuple[str, str]] = []
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"Empty building-block file: {path}")
    logger.info("Read %d lines from %s", len(lines), path)

    for i, line in enumerate(tqdm(lines, desc="Validate BB file", leave=False), start=1):
        fields = line.strip().split("\t")
        if not (len(fields) == 2 and fields[0] and fields[1]):
            raise ValueError(f"Invalid format in line {i}: {line.strip()}")
        if ";" in fields[1]:
            raise ValueError(f"Building-block ID must not contain semicolons (line {i})")
        records.append((fields[0], fields[1]))

    ids = {identifier for _, identifier in records}
    if len(ids) != len(records):
        raise ValueError("Duplicate building-block identifiers found")
    logger.info("Read %d building blocks.", len(records))
    return records


def _clean_building_blocks(
    records: list[tuple[str, str]],
    allow_isotope: bool = False,
    remove_salts: bool = True,
) -> list[tuple[str, str]]:
    """Canonicalize BBs and retain allowed, single-component structures."""
    salt_remover = SaltRemover()

    def _canonicalize(smiles: str) -> str | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not mol.GetNumAtoms():
            return None
        if remove_salts:
            mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
            if mol is None or not mol.GetNumAtoms():
                return None
        if not all(atom.GetSymbol() in ALLOWED_ATOMIC_TYPES for atom in mol.GetAtoms()):
            return None
        if not allow_isotope and any(atom.GetIsotope() for atom in mol.GetAtoms()):
            return None
        canonical_smi = Chem.MolToSmiles(mol)
        if "." in canonical_smi:
            return None
        return canonical_smi

    cleaned_records = []
    for smiles, identifier in tqdm(records, desc="Cleaning BBs", leave=False):
        canonical_smi = _canonicalize(smiles)
        if canonical_smi is not None:
            cleaned_records.append((canonical_smi, identifier))
    if not cleaned_records:
        raise ValueError("No valid building blocks")
    logger.info(
        "Cleaned building blocks: %d valid / %d total", len(cleaned_records), len(records)
    )
    return cleaned_records


def read_blocks(
    building_blocks: str | Path,
    env_dir: str | Path,
    druglikeness_threshold: float | None = None,
    druglikeness_device: str = "cpu",
) -> None:
    """Clean and optionally screen BBs; save ID-to-SMILES provenance for conversion."""
    env_path = Path(env_dir)
    source_path = Path(building_blocks)
    records = _read_building_blocks(source_path)
    records = _clean_building_blocks(records)

    # Screen cleaned source molecules before converting them to synthons.
    if druglikeness_threshold is not None:
        from druglikeness.deepdl import DeepDL

        num_input = len(records)
        threshold, device = druglikeness_threshold, druglikeness_device
        if not 0 <= threshold <= 100:
            raise ValueError("Druglikeness threshold must be between 0 and 100")
        logger.info("Loading DeepDL: device=%s, threshold=%s", device, threshold)
        model = DeepDL.from_pretrained(device=device)
        batch_size = 256 if device.startswith("cuda") else 64
        smi_list = [smiles for smiles, _ in records]
        scores = model.screening(
            smi_list, naive=True, batch_size=batch_size, verbose=True
        )
        records = [
            record
            for record, score in zip(records, scores, strict=True)
            if score >= threshold
        ]
        logger.info(
            "Druglikeness: retained %d / %d building blocks", len(records), num_input
        )
        if not records:
            raise ValueError("No building blocks passed druglikeness filtering")
        del model

    # Stable source order keeps preparation independent of input row order.
    records.sort(key=lambda x: x[1])

    env_path.mkdir(parents=True, exist_ok=True)
    bb_smi_path = env_path / "building_blocks.smi"
    with bb_smi_path.open("w") as w:
        for smiles, identifier in records:
            w.write(f"{smiles}\t{identifier}\n")
