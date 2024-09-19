import numpy as np
import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import QED

from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet.config import Config
from gflownet.trainer import FlatRewards
from gflownet.misc.chem_metrics import mol2qed, mol2sascore, mol2vina

from gflownet.base.base_task import BaseTask, BaseMOOTask


aux_tasks = {"qed": mol2qed, "sa": mol2sascore, "vina": mol2vina}


class UniDockTask(BaseTask):
    """Sets up a task where the reward is computed using a UniDock - TacoGFN Task."""

    def __init__(self, cfg: Config, rng: np.random.Generator, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, rng, wrap_model)
        self.protein_path: str = cfg.task.docking.protein_path
        self.center: tuple[float, float, float] = cfg.task.docking.center
        self.best_molecules: list[tuple[float, str]] = []

    def compute_flat_rewards(self, mols: list[RDMol], batch_idx: list[int]) -> tuple[FlatRewards, Tensor]:
        vina_scores = mol2vina(mols, self.protein_path, self.center).reshape(-1, 1)
        is_valid_t = torch.ones((len(mols),), dtype=torch.bool)
        return FlatRewards(vina_scores), is_valid_t

    def update_best_molecules(self, mols: list[RDMol], scores: list[float]):
        best_smi = [smi for score, smi in self.best_molecules]
        score_smiles = [
            (score, Chem.MolToSmiles(mol)) for score, mol in zip(scores, mols, strict=True) if self.constraint(mol)
        ]
        score_smiles = [(score, smi) for score, smi in score_smiles if smi not in best_smi]
        self.best_molecules = sorted(self.best_molecules + score_smiles, reverse=True)[:100]

    def constraint(self, mol: RDMol) -> bool:
        return True


class UniDockMOOTask(UniDockTask, BaseMOOTask):
    """Sets up a task where the reward is computed using a UniDock, QED, SAScore."""

    def __init__(self, cfg: Config, rng: np.random.Generator, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, rng, wrap_model)
        assert set(self.objectives) <= {"vina", "qed", "sa"}

    def compute_flat_rewards(self, mols: list[RDMol], batch_idx: list[int]) -> tuple[FlatRewards, Tensor]:
        is_valid_t = torch.ones(len(mols), dtype=torch.bool)

        fr: Tensor
        flat_r: list[Tensor] = []
        self.avg_reward_info = []
        for obj in self.objectives:
            if obj == "vina":
                fr = aux_tasks[obj](mols, self.protein_path, self.center)
                self.update_best_molecules(mols, fr.tolist())
                flat_r.append(fr * 0.1)
            else:
                fr = aux_tasks[obj](mols)
                flat_r.append(fr)
            self.avg_reward_info.append((obj, fr.mean().item()))
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(mols)
        return FlatRewards(flat_rewards), is_valid_t

    def constraint(self, mol: RDMol) -> bool:
        return QED.qed(mol) > 0.5
