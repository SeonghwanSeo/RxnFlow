from pathlib import Path
import numpy as np
import torch

from rdkit import Chem, DataStructs
from rdkit.Chem import QED

from rdkit.Chem import Mol as RDMol

from gflownet.utils import sascore
from .unidock import unidock_scores


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


def mol2vina(
    mols: list[RDMol],
    protein_path: str | Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float] = (22.5, 22.5, 22.5),
    search_mode: str = "balance",
    out_dir: Path | str | None = None,
):
    vina_score = unidock_scores(mols, protein_path, center, size, seed=1, search_mode=search_mode, out_dir=out_dir)
    return torch.tensor(vina_score, dtype=torch.float).clip(max=0.0)
