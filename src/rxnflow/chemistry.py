"""RDKit parsing and molecular/building-block descriptors."""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, rdFingerprintGenerator
from torch import Tensor

PROPERTY_DIM = 9
FINGERPRINT_DIM = 678
PROPERTY_NAMES = (
    "mw",
    "tpsa",
    "hbd",
    "hba",
    "logp",
    "rotatable_bonds",
    "rings",
    "aromatic_rings",
    "heavy_atoms",
)
PROPERTY_SCALE = (500.0, 150.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 50.0)


def parse_molecule(smiles: str) -> Chem.Mol | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.RemoveHs(mol)


def heavy_atom_count(mol: Chem.Mol | None) -> int:
    """Count heavy atoms, including non-hydrogen attachment markers."""

    if mol is None:
        return 0
    return int(mol.GetNumHeavyAtoms())


def molecular_properties(mol: Chem.Mol | None) -> Tensor:
    if mol is None:
        return torch.zeros(PROPERTY_DIM, dtype=torch.float32)
    values = [
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.MolLogP(mol),
        Lipinski.NumRotatableBonds(mol),
        Lipinski.RingCount(mol),
        Lipinski.NumAromaticRings(mol),
        heavy_atom_count(mol),
    ]
    return torch.tensor(values, dtype=torch.float32)


def normalize_molecular_properties(values: Tensor) -> Tensor:
    """Scale molecular properties using a tensor local to the input device and dtype."""

    return values / values.new_tensor(PROPERTY_SCALE)


def block_fingerprint(mol: Chem.Mol) -> Tensor:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    morgan = generator.GetCountFingerprint(mol)
    morgan_array = np.zeros(512, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(morgan, morgan_array)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_array = np.zeros(167, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(maccs, maccs_array)
    return torch.from_numpy(np.concatenate([morgan_array, maccs_array[1:]]))


def block_feature_row(smiles: str) -> tuple[Tensor, Tensor, int]:
    mol = parse_molecule(smiles)
    if mol is None:
        raise ValueError(f"invalid building-block SMILES: {smiles}")
    return molecular_properties(mol), block_fingerprint(mol), heavy_atom_count(mol)
