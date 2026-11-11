"""RDKit parsing and features for atoms, bonds, molecules and synthons."""

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

from rxnflow.core.synthon import SYNTHON_TYPE_LIMIT


def _get_index(value: object, choices: tuple, *, allow_unk: bool) -> int:
    try:
        return choices.index(value)
    except ValueError:
        if allow_unk:
            return len(choices)
        raise ValueError(f"Unknown feature value {value!r}") from None


# === Atom features === #
ATOM_FEATURE_DICT = {
    "type": (0, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53),
    "degree": (0, 1, 2, 3, 4, 5),
    "charge": (-2, -1, 0, 1, 2),
    "num_hs": (0, 1, 2, 3, 4),
    "hybridization": (
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
    ),
    "chiral": (
        Chem.ChiralType.CHI_UNSPECIFIED,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ),
}
NODE_DIM = (
    sum(len(choices) + 1 for choices in ATOM_FEATURE_DICT.values())
    + 2  # aromaticity and mass
    + SYNTHON_TYPE_LIMIT
)


def get_atom_features(atom: Chem.Atom, out: np.ndarray | None = None) -> np.ndarray:
    """Encode one atom, overwriting the supplied row when given."""
    if out is None:
        out = np.empty(NODE_DIM, dtype=np.float32)
    out.fill(0)

    atomic_num = atom.GetAtomicNum()
    is_attachment = atomic_num == 0

    atom_values = {
        "type": atomic_num,
        "degree": atom.GetDegree(),
        "charge": atom.GetFormalCharge(),
        "num_hs": atom.GetTotalNumHs(includeNeighbors=True),
        "hybridization": atom.GetHybridization(),
        "chiral": atom.GetChiralTag(),
    }
    offset = 0
    for field, choices in ATOM_FEATURE_DICT.items():
        index = _get_index(atom_values[field], choices, allow_unk=True)
        out[offset + index] = 1
        offset += len(choices) + 1

    attachment_start = offset + 2
    out[offset:attachment_start] = (
        atom.GetIsAromatic(),
        atom.GetMass() / 100.0 if not is_attachment else 0.0,
    )
    if is_attachment:
        label = atom.GetIsotope()
        if not 1 <= label <= SYNTHON_TYPE_LIMIT:
            raise ValueError(
                f"Invalid attachment label {label}. Expected 1–{SYNTHON_TYPE_LIMIT}."
            )
        out[attachment_start + (label - 1)] = 1
    return out


# === Bond features === #
BOND_FEATURE_DICT = {
    "bond_type": (
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ),
    "stereo": (
        Chem.BondStereo.STEREONONE,
        Chem.BondStereo.STEREOANY,
        Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS,
        Chem.BondStereo.STEREOTRANS,
    ),
}
EDGE_DIM = (
    len(BOND_FEATURE_DICT["bond_type"])
    + (len(BOND_FEATURE_DICT["stereo"]) + 1)
    + 2  # conjugation and ring
)


def get_bond_features(bond: Chem.Bond, out: np.ndarray | None = None) -> np.ndarray:
    """Encode one bond, overwriting the supplied row when given."""
    if out is None:
        out = np.empty(EDGE_DIM, dtype=np.float32)
    out.fill(0)

    bond_types = BOND_FEATURE_DICT["bond_type"]
    bond_type_index = _get_index(bond.GetBondType(), bond_types, allow_unk=False)
    out[bond_type_index] = 1

    stereo_types = BOND_FEATURE_DICT["stereo"]
    stereo_index = _get_index(bond.GetStereo(), stereo_types, allow_unk=True)
    out[len(bond_types) + stereo_index] = 1
    out[-2:] = bond.GetIsConjugated(), bond.IsInRing()
    return out


# === Molecule features === #

PROPERTY_DIM = 9
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


def get_mol_features(mol: Chem.Mol | None, out: np.ndarray | None = None) -> np.ndarray:
    """Write raw descriptors in PROPERTY_NAMES order, or zeros for the empty state."""
    if out is None:
        out = np.empty(PROPERTY_DIM, dtype=np.float32)
    if mol is None:
        out.fill(0)
        return out
    out[:] = (
        Descriptors.ExactMolWt(mol),
        Descriptors.TPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.MolLogP(mol),
        Lipinski.NumRotatableBonds(mol),
        Lipinski.RingCount(mol),
        Lipinski.NumAromaticRings(mol),
        mol.GetNumHeavyAtoms(),
    )
    return out


# === Synthon features === #
FINGERPRINT_DIM = 512 + 166
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)


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


def get_synthon_features(smiles: str) -> tuple[np.ndarray, np.ndarray, int]:
    mol = parse_molecule(smiles)
    if mol is None:
        raise ValueError(f"invalid synthon SMILES: {smiles}")
    return get_mol_features(mol), synthon_fingerprint(mol), heavy_atom_count(mol)
