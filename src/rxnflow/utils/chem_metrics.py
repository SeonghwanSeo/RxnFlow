import numpy as np
import torch

from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Descriptors
from rdkit.Chem import Mol as RDMol

from gflownet.utils import sascore


def compute_diverse_top_k(
    smiles: list[str],
    rewards: list[float],
    k: int,
    thresh: float = 0.5,
) -> list[int]:
    modes = [(i, smi, float(r)) for i, (r, smi) in enumerate(zip(rewards, smiles, strict=True))]
    modes.sort(key=lambda m: m[2], reverse=True)
    top_modes = [modes[0][0]]

    prev_smis = {modes[0][1]}
    mode_fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(modes[0][1]))]
    for i in range(1, len(modes)):
        smi = modes[i][1]
        if smi in prev_smis:
            continue
        prev_smis.add(smi)
        if thresh > 0:
            fp = Chem.RDKFingerprint(Chem.MolFromSmiles(smi))
            sim = DataStructs.BulkTanimotoSimilarity(fp, mode_fps)
            if max(sim) >= thresh:  # div = 1- sim
                continue
            mode_fps.append(fp)
            top_modes.append(modes[i][0])
        else:
            top_modes.append(modes[i][0])
        if len(top_modes) >= k:
            break
    return top_modes


def calc_diversity(smiles_list: list[str]):
    x = [Chem.RDKFingerprint(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]
    s = np.array([DataStructs.BulkTanimotoSimilarity(i, x) for i in x])
    n = s.shape[0]
    return 1 - (np.sum(s) - n) / (n**2 - n)


def safe(f, x, default):
    try:
        return f(x)
    except Exception:
        return default


def mol2sascore(mols: list[RDMol], default=10):
    sas = torch.tensor([safe(sascore.calculateScore, mol, default) for mol in mols])
    sas = (10 - sas) / 9  # Turn into a [0-1] reward
    return sas


def mol2qed(mols: list[RDMol], default=0):
    return torch.tensor([safe(QED.qed, mol, default) for mol in mols])


def mol2mw(mols: list[RDMol]):
    return torch.tensor([Descriptors.MolWt(mol) for mol in mols])
