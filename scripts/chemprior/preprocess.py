"""Prepare clean, tokenizable .smi files for ChemPrior fine-tuning."""

import argparse
import json
from collections.abc import Iterator
from multiprocessing import get_context
from pathlib import Path

from rdkit import Chem

from chemprior.cleaning import ParentCleaner
from chemprior.tokenizer import SmilesTokenizer


def _read_smiles(path: str | Path) -> Iterator[tuple[str, str]]:
    """Read SMILES and optional ID from each data row."""
    with Path(path).expanduser().open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line and not line.startswith("#"):
                fields = line.split(maxsplit=2)
                yield fields[0], fields[1] if len(fields) > 1 else ""


def _canonicalize(smiles: str, cleaner: ParentCleaner) -> str | None:
    """Return a canonical parent, or None for invalid or ambiguous molecules."""
    try:
        molecule = cleaner.clean(smiles)
        if molecule is None:
            return None
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except ValueError:
        return None


def _init_worker(max_length):
    global _cleaner, _tokenizer, _max_length
    _cleaner = ParentCleaner()
    _tokenizer = SmilesTokenizer()
    _max_length = max_length


def _prepare_one(row):
    raw, identifier = row
    smiles = _canonicalize(raw, _cleaner)
    if smiles is None:
        return "invalid", None, identifier
    try:
        tokens = _tokenizer.encode(smiles, add_special_tokens=False)
    except (KeyError, ValueError, AttributeError):
        return "unsupported", None, identifier
    if len(tokens) > _max_length:
        return "too_long", None, identifier
    return "accepted", smiles, identifier


def preprocess_smiles(path, output, max_length=255, num_workers=1):
    """Clean in input order and deduplicate canonical isomeric SMILES."""
    if not 1 <= max_length <= 2047 or num_workers < 1:
        raise ValueError("Invalid max_length or num_workers")
    counts = dict(rows=0, accepted=0, invalid=0, unsupported=0, too_long=0, duplicate=0)
    seen = set()
    pool = None
    if num_workers == 1:
        _init_worker(max_length)
        results = map(_prepare_one, _read_smiles(path))
    else:
        pool = get_context("spawn").Pool(num_workers, _init_worker, (max_length,))
        results = pool.imap(_prepare_one, _read_smiles(path), chunksize=256)
    try:
        with Path(output).open("x", encoding="utf-8") as out:
            for status, smiles, identifier in results:
                counts["rows"] += 1
                if status == "accepted" and smiles in seen:
                    status = "duplicate"
                counts[status] += 1
                if status == "accepted":
                    seen.add(smiles)
                    out.write(smiles + ("\t" + identifier if identifier else "") + "\n")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if not counts["accepted"]:
        raise ValueError(f"No usable molecules: {counts}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input .smi file")
    parser.add_argument("--output", required=True, help="Output cleaned .smi file")
    parser.add_argument("--max-length", type=int, default=255)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    counts = preprocess_smiles(args.input, args.output, args.max_length, args.num_workers)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
