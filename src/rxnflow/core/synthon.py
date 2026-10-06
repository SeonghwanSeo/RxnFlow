"""Typed synthon definitions and conversion, independent of environment I/O."""

from __future__ import annotations

from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts


class SynthonConversion:
    def __init__(self, synthon_type: int, original: str, converted: str):
        self.synthon_type = synthon_type
        template = f"{original}>>{converted}"
        reaction = ReactionFromSmarts(template)
        if reaction is None:
            raise ValueError(f"invalid synthon conversion {synthon_type}: {template}")
        reaction.Initialize()
        self.reaction: ChemicalReaction = reaction
        self.product_pattern = reaction.GetProductTemplate(0)

    def convert(self, mol: Chem.Mol) -> list[str]:
        """Return unique canonical SMILES of sanitized, pattern-matching products."""
        products: list[str] = []
        for product_tuple in self.reaction.RunReactants((mol,), 0):
            if len(product_tuple) != 1:
                continue
            product = product_tuple[0]
            try:
                Chem.SanitizeMol(product)
                product = Chem.RemoveHs(product)
                # Sanitization can change aromaticity or H counts at the handle.
                if not product.HasSubstructMatch(self.product_pattern):
                    continue
                smiles = Chem.MolToSmiles(product)
            except (ValueError, RuntimeError, Chem.rdchem.KekulizeException):
                continue
            if "." in smiles:
                continue
            products.append(smiles)
        return sorted(set(products))


def get_dummy_atoms(mol: Chem.Mol) -> tuple[Chem.Atom, ...]:
    """Return dummy atoms."""
    return tuple(atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)


def typed_dummy_isotopes(mol: Chem.Mol) -> tuple[int, ...]:
    """Return sorted dummy labels."""
    return tuple(sorted(atom.GetIsotope() for atom in get_dummy_atoms(mol)))


def load_synthon_templates(path: Path) -> dict[int, tuple[str, str]]:
    """Read conversion patterns keyed by synthon type."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a list")
    templates: dict[int, tuple[str, str]] = {}
    types: list[int] = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict) or set(value) != {
            "type",
            "original",
            "convert",
        }:
            raise ValueError(f"{path}: invalid synthon entry {index + 1}")
        synthon_type = value["type"]
        if not 1 <= synthon_type < 100:
            raise ValueError(f"invalid synthon type: {synthon_type}")
        templates[synthon_type] = (value["original"], value["convert"])
        types.append(synthon_type)
    if not types or types != sorted(set(types)):
        raise ValueError("synthon.yaml must define unique types in increasing order")
    return templates
