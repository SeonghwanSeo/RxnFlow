import numpy as np
from numpy.typing import NDArray

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

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
        feature_out = np.empty(BLOCK_PROPERTY_DIM, dtype=np.float32)
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

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_fp_arr = np.array(maccs_fp)[:166]
    fp_out[:166] = maccs_fp_arr / min(np.linalg.norm(maccs_fp_arr, 2), 1)

    mg = GetMorganGenerator(FP_RADIUS, fpSize=FP_NBITS)
    morgan_fp_arr = mg.GetFingerprintAsNumPy(mol)
    fp_out[166:] = morgan_fp_arr / min(np.linalg.norm(morgan_fp_arr, 2), 1)
    return fp_out, feature_out
