"""RDKit parsing and molecular/building-block descriptors."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors,
    Lipinski,
    MACCSkeys,
    rdFingerprintGenerator,
    rdMolDescriptors,
)

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
_PROPERTY_SCALE = [500.0, 150.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 50.0]
PROPERTY_SCALE = np.array(_PROPERTY_SCALE, dtype=np.float32)


def parse_molecule(smiles: str) -> Chem.Mol | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.RemoveHs(mol)


def heavy_atom_count(mol: Chem.Mol | None) -> int:
    """RDKit heavy-atom count; hydrogens and dummy handles are excluded."""

    if mol is None:
        return 0
    return int(mol.GetNumHeavyAtoms())


def molecular_properties(mol: Chem.Mol | None) -> np.ndarray:
    if mol is None:
        return np.zeros(PROPERTY_DIM, dtype=np.float32)
    # A dummy isotope is a categorical synthesis label, not an isotope mass.
    # Descriptors still describe the abstract synthon; no hidden real molecule
    # is reconstructed. Terminal products contain no dummies and need no change.
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        mol = Chem.Mol(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)
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
    return np.array(values, dtype=np.float32)


def normalize_molecular_properties(values: np.ndarray) -> np.ndarray:
    """Scale NumPy descriptors for graph input; model tensors normalize on device."""

    return values / PROPERTY_SCALE


def block_fingerprint(mol: Chem.Mol) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    invariants = rdMolDescriptors.GetConnectivityInvariants(mol)
    # Default Morgan invariants ignore dummy isotopes. Supply the categorical
    # label explicitly, so active/latent types and their positions remain
    # visible to both the block and outcome encoders (without fake atom masses).
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            invariants[atom.GetIdx()] = atom.GetIsotope()
    morgan = generator.GetCountFingerprint(mol, customAtomInvariants=invariants)
    morgan_array = np.zeros(512, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(morgan, morgan_array)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_array = np.zeros(167, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(maccs, maccs_array)
    return np.concatenate([morgan_array, maccs_array[1:]])


def block_feature_row(smiles: str) -> tuple[np.ndarray, np.ndarray, int]:
    mol = parse_molecule(smiles)
    if mol is None:
        raise ValueError(f"invalid building-block SMILES: {smiles}")
    return molecular_properties(mol), block_fingerprint(mol), heavy_atom_count(mol)
