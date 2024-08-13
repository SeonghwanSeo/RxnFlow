import numpy as np
import torch

from rdkit import Chem, DataStructs
from rdkit.Chem import QED

from rdkit.Chem import Mol as RDMol

from gflownet.utils import sascore
from gflownet.misc.unidock.utils import unidock_scores


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


def mol2vina(mols: list[RDMol], protein_path: str, center: tuple[float, float, float]):
    vina_score = unidock_scores(mols, protein_path, center)
    return torch.tensor(vina_score, dtype=torch.float).neg().clip(min=0.0)
