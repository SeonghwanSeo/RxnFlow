"""Normalized molecular samples exposed to rewards and user filters."""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem


@dataclass(frozen=True)
class Sample:
    """A valid molecular sample with canonical SMILES and an RDKit molecule."""

    smiles: str
    mol: Chem.Mol = field(repr=False, compare=False)

    @classmethod
    def from_smiles(cls, smiles: str) -> Sample | None:
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return cls.from_mol(mol)

    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> Sample | None:
        normalized = Chem.RemoveHs(Chem.Mol(mol))
        smiles = Chem.MolToSmiles(normalized, canonical=True)
        if not smiles:
            return None
        return cls(smiles=smiles, mol=normalized)


SampleInput = Sample | Chem.Mol | str | None


def as_sample(value: SampleInput) -> Sample | None:
    """Normalize a public sample input without burdening reward authors."""

    if value is None:
        return None
    if isinstance(value, Sample):
        return value
    if isinstance(value, str):
        return Sample.from_smiles(value)
    if isinstance(value, Chem.Mol):
        return Sample.from_mol(value)
    raise TypeError(f"unsupported sample type: {type(value).__name__}")
