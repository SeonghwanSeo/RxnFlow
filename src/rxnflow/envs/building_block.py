import numpy as np
from numpy.typing import NDArray

from rdkit import Chem
from rdkit.Chem import Crippen, MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

FP_RADIUS = 2
FP_NBITS = 1024
BLOCK_FP_DIM = 1024 + 166
BLOCK_PROPERTY_DIM = 8
NUM_BLOCK_FEATURES = BLOCK_FP_DIM + BLOCK_PROPERTY_DIM


def get_block_features(mol: str | Chem.Mol) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
    """Setup Building Block Datas"""
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    # NOTE: MACCS Fingerprint, Morgan Fingerprint
    maccs_fp = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.bool_)[:166]
    mg = GetMorganGenerator(FP_RADIUS, fpSize=FP_NBITS)
    morgan_fp = mg.GetFingerprintAsNumPy(mol).astype(np.bool_)
    fp_out = np.concatenate([maccs_fp, morgan_fp])

    # NOTE: Common RDKit Descriptors
    feature = []
    feature.append(rdMolDescriptors.CalcExactMolWt(mol) / 100)
    feature.append(rdMolDescriptors.CalcNumHeavyAtoms(mol) / 10)
    feature.append(rdMolDescriptors.CalcNumHBA(mol) / 10)
    feature.append(rdMolDescriptors.CalcNumHBD(mol) / 10)
    feature.append(rdMolDescriptors.CalcNumAromaticRings(mol) / 10)
    feature.append(rdMolDescriptors.CalcNumAliphaticRings(mol) / 10)
    feature.append(rdMolDescriptors.CalcTPSA(mol) / 100)
    feature.append(Crippen.MolLogP(mol) / 10)
    feature_out = np.array(feature, dtype=np.float32)

    return fp_out, feature_out
