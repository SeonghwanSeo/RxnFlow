from pathlib import Path
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


aux_tasks = {"qed": mol2qed, "sa": mol2sascore}


class UniDockTask(BaseTask):
    """Sets up a task where the reward is computed using a UniDock - TacoGFN Task."""

    def __init__(self, cfg: Config, rng: np.random.Generator, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, rng, wrap_model)
        self.protein_path: str = cfg.task.docking.protein_path
        self.center: tuple[float, float, float] = cfg.task.docking.center
        self.last_molecules: list[tuple[float, str]] = []
        self.best_molecules: list[tuple[float, str]] = []
        self.save_dir: Path = Path(cfg.log_dir) / "unidock"
        self.oracle_idx = 0
        self.search_mode: str = "balance"

    def compute_flat_rewards(self, mols: list[RDMol], batch_idx: list[int]) -> tuple[FlatRewards, Tensor]:
        is_valid_t = torch.ones((len(mols),), dtype=torch.bool)
        vina_scores = self.run_vina(mols)
        self.update_best_molecules(mols, vina_scores.tolist())
        fr = vina_scores.neg().reshape(-1, 1)
        return FlatRewards(fr), is_valid_t

    def update_best_molecules(self, mols: list[RDMol], scores: list[float]):
        best_smi = [smi for score, smi in self.best_molecules]
        score_smiles = [
            (score, Chem.MolToSmiles(mol)) for score, mol in zip(scores, mols, strict=True) if self.constraint(mol)
        ]
        score_smiles = [(score, smi) for score, smi in score_smiles if smi not in best_smi]
        self.last_molecules = score_smiles
        self.best_molecules = sorted(self.best_molecules + score_smiles, reverse=False)[:100]

    def run_vina(self, mols: list[RDMol]) -> Tensor:
        out_dir = self.save_dir / f"oracle{self.oracle_idx}"
        vina_score = mol2vina(mols, self.protein_path, self.center, self.search_mode, out_dir)
        self.oracle_idx += 1
        return vina_score

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
                vina_scores = self.run_vina(mols)
                self.update_best_molecules(mols, vina_scores.tolist())
                fr = vina_scores * -0.1
                flat_r.append(fr)
            else:
                fr = aux_tasks[obj](mols)
                flat_r.append(fr)
            self.avg_reward_info.append((obj, fr.mean().item()))
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(mols)
        return FlatRewards(flat_rewards), is_valid_t

    def constraint(self, mol: RDMol) -> bool:
        return QED.qed(mol) > 0.5
