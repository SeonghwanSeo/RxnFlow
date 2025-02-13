import numpy as np
from rdkit import Chem, DataStructs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from . import sascore


def compute_diverse_top_k(
    smiles: list[str],
    rewards: list[float],
    k: int,
    thresh: float = 0.5,
) -> list[tuple[int, str, float]]:
    modes = [(i, smi, r) for i, (r, smi) in enumerate(zip(rewards, smiles, strict=True))]
    modes.sort(key=lambda m: m[2], reverse=True)
    top_modes = [modes[0]]

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
            if max(sim) >= (1 - thresh):  # div = 1- sim
                continue
            mode_fps.append(fp)
            top_modes.append(modes[i])
        else:
            top_modes.append(modes[i])
        if len(top_modes) >= k:
            break
    return top_modes


def calculate_diversity(smiles: list[str]):
    x = [Chem.RDKFingerprint(Chem.MolFromSmiles(smi)) for smi in smiles]
    s = np.array([DataStructs.BulkTanimotoSimilarity(i, x) for i in x])
    x = s.shape[0]
    return 1 - (np.sum(s) - x) / (x**2 - x)
