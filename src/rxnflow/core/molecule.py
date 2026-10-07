"""An RDKit molecule paired with its SMILES."""

from __future__ import annotations

from rdkit import Chem


class Molecule:
    """Treat smiles and rdmol as read-only; equality and hash use SMILES."""

    smiles: str
    rdmol: Chem.Mol

    def __init__(self, smiles: str | None = None, rdmol: Chem.Mol | None = None) -> None:
        if smiles is None and rdmol is None:
            raise ValueError("smiles or rdmol is required")
        if rdmol is None:
            rdmol = Chem.MolFromSmiles(smiles)
            if rdmol is None:
                raise ValueError(f"invalid molecule SMILES: {smiles}")
        if smiles is None:
            smiles = Chem.MolToSmiles(rdmol)
        self.smiles = smiles
        self.rdmol = rdmol

    def __repr__(self) -> str:
        return self.smiles

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Molecule):
            return NotImplemented
        return self.smiles == other.smiles

    def __hash__(self) -> int:
        return hash(self.smiles)
