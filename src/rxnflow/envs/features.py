"""RDKit parsing and molecular/synthon descriptors."""

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

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)

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
# logP is not additive; heavy-atom capacity is controlled by generation.max_atoms.
PROPERTY_PENALTY_INDICES = {
    name: index
    for index, name in enumerate(PROPERTY_NAMES)
    if name not in ("logp", "heavy_atoms")
}
PROPERTY_SCALE_DICT = {
    "mw": 100.0,
    "tpsa": 100.0,
    "hbd": 10.0,
    "hba": 10.0,
    "logp": 10.0,
    "rotatable_bonds": 10.0,
    "rings": 10.0,
    "aromatic_rings": 10.0,
    "heavy_atoms": 100.0,
}
_PROPERTY_SCALE = [PROPERTY_SCALE_DICT[name] for name in PROPERTY_NAMES]
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
    """Return raw descriptors in PROPERTY_NAMES order; the empty state is zero."""
    if mol is None:
        return np.zeros(PROPERTY_DIM, dtype=np.float32)
    # Dummy isotopes encode synthesis types, so exclude their labels from mass.
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        mol = Chem.Mol(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)
    values = [
        Descriptors.ExactMolWt(mol),
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


def synthon_fingerprint(mol: Chem.Mol) -> np.ndarray:
    """Concatenate isotope-aware Morgan counts and MACCS bits into 678 bytes."""
    invariants = rdMolDescriptors.GetConnectivityInvariants(mol)
    # Default Morgan invariants ignore dummy isotopes. Supply the categorical
    # label explicitly, so active/latent types and their positions remain
    # visible to the synthon encoder (without fake atom masses).
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            invariants[atom.GetIdx()] = atom.GetIsotope()
    morgan = _MORGAN_GENERATOR.GetCountFingerprint(mol, customAtomInvariants=invariants)
    # Saturate counts before uint8 conversion to prevent overflow.
    morgan_array = np.zeros(512, dtype=np.uint32)
    DataStructs.ConvertToNumpyArray(morgan, morgan_array)
    np.minimum(morgan_array, 255, out=morgan_array)
    # MACCS index 0 is unused; concatenate only its 166 defined keys.
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_array = np.zeros(167, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(maccs, maccs_array)
    return np.concatenate([morgan_array.astype(np.uint8), maccs_array[1:]])


def synthon_feature_row(smiles: str) -> tuple[np.ndarray, np.ndarray, int]:
    mol = parse_molecule(smiles)
    if mol is None:
        raise ValueError(f"invalid synthon SMILES: {smiles}")
    return molecular_properties(mol), synthon_fingerprint(mol), heavy_atom_count(mol)
