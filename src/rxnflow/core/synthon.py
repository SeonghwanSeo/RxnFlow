"""Typed synthon definitions and conversion, independent of environment I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts


@dataclass(frozen=True)
class SynthonSpec:
    type: int
    original: str
    convert: str


class SynthonConversion:
    def __init__(self, spec: SynthonSpec):
        self.spec = spec
        template = f"{spec.original}>>{spec.convert}"
        reaction = ReactionFromSmarts(template)
        if reaction is None:
            raise ValueError(f"invalid synthon conversion {spec.type}: {template}")
        reaction.Initialize()
        self.reaction: ChemicalReaction = reaction
        self.product_pattern = reaction.GetProductTemplate(0)

    def run_mol(self, mol: Chem.Mol) -> dict[str, Chem.Mol]:
        """Return sanitized conversions matching the convert pattern, by SMILES."""
        products: dict[str, Chem.Mol] = {}
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
            if "." not in smiles:
                products.setdefault(smiles, product)
        # Reuse sanitized products during conversion and site inspection.
        return {smiles: products[smiles] for smiles in sorted(products)}


def typed_dummy_isotopes(mol: Chem.Mol) -> tuple[int, ...]:
    """Return sorted dummy labels, including isotope-0 catalog attachments."""
    return tuple(
        sorted(atom.GetIsotope() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
    )


def load_synthon_specs(path: Path) -> list[SynthonSpec]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a list")
    specs: list[SynthonSpec] = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict) or set(value) != {
            "type",
            "original",
            "convert",
        }:
            raise ValueError(f"{path}: invalid synthon entry {index + 1}")
        spec = SynthonSpec(**value)
        if not 1 <= spec.type < 100:
            raise ValueError(f"invalid synthon type: {spec.type}")
        specs.append(spec)
    types = [spec.type for spec in specs]
    if not types or types != sorted(set(types)):
        raise ValueError("synthon.yaml must define unique types in increasing order")
    return specs
