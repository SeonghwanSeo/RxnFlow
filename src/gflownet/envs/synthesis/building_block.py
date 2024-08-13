import numpy as np
from numpy.typing import NDArray

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors


def get_block_features(
    mol: str | Chem.Mol,
    fp_radius: int,
    fp_nbits: int,
    fp_out: NDArray | None = None,
    feature_out: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    # NOTE: Setup Building Block Datas

    if fp_out is None:
        fp_out = np.empty(166 + fp_nbits, dtype=np.bool_)
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

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    fp_out[:166] = np.array(maccs_fp)[:166]
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, fp_nbits)
    fp_out[166:] = np.frombuffer(morgan_fp.ToBitString().encode(), "u1") - ord("0")
    return fp_out, feature_out
