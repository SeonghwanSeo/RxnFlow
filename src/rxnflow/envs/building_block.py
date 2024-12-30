import numpy as np
from numpy.typing import NDArray

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors

FP_RADIUS = 2
FP_NBITS = 1024
BLOCK_FP_DIM = FP_NBITS + 166
BLOCK_PROPERTY_DIM = 8
NUM_BLOCK_FEATURES = BLOCK_FP_DIM + BLOCK_PROPERTY_DIM


def get_block_features(
    mol: str | Chem.Mol,
    fp_out: NDArray | None = None,
    feature_out: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    # NOTE: Setup Building Block Datas

    if fp_out is None:
        fp_out = np.empty(BLOCK_FP_DIM, dtype=np.bool_)
        assert fp_out is not None

    if feature_out is None:
        feature_out = np.empty(8, dtype=np.float32)
        assert feature_out is not None

    # NOTE: Common RDKit Descriptors
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    feature_out[0] = rdMolDescriptors.CalcExactMolWt(mol) / 100
    feature_out[1] = rdMolDescriptors.CalcNumHeavyAtoms(mol) / 10
    feature_out[2] = rdMolDescriptors.CalcNumHBA(mol) / 10
    feature_out[3] = rdMolDescriptors.CalcNumHBD(mol) / 10
    feature_out[4] = rdMolDescriptors.CalcNumAromaticRings(mol) / 10
    feature_out[5] = rdMolDescriptors.CalcNumAliphaticRings(mol) / 10
    feature_out[6] = Descriptors.MolLogP(mol) / 10
    feature_out[7] = Descriptors.TPSA(mol) / 100
    feature_out *= 10  # NOTE: fingerprint: 1024+166 dim, feature: 8 dim.

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    fp_out[:166] = np.array(maccs_fp)[:166]
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, FP_NBITS)
    fp_out[166:] = np.frombuffer(morgan_fp.ToBitString().encode(), "u1") - ord("0")
    return fp_out, feature_out
