import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Lipinski

from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet.config import Config
from gflownet.trainer import FlatRewards

from gflownet.base.base_trainer import SynthesisTrainer
from gflownet.base.base_task import BaseTask
from unidock_tools.modules.protein_prep.pdb2pdbqt import pdb2pdbqt

from gflownet.misc.unidock.scoring_concise import unidock_scores
from gflownet.misc.extend_3d.env_context import Synthesis3DEnvContext
from gflownet.misc.extend_3d import gfn


class UniDockLipinskiTask(BaseTask):
    """Sets up a task where the reward is computed using a UniDock - TacoGFN Task."""

    def __init__(self, cfg: Config, rng: np.random.Generator, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, rng, wrap_model)
        self.protein_path: str = os.path.join(cfg.log_dir, "protein.pdbqt")
        pdb2pdbqt(cfg.task.docking.protein_path, self.protein_path)

        self.center: tuple[float, float, float] = cfg.task.docking.center
        self.last_molecules: list[tuple[float, str]] = []
        self.best_molecules: list[tuple[float, str]] = []
        self.save_dir: Path = Path(cfg.log_dir) / "unidock"
        self.oracle_idx = 0
        self.search_mode: str = "balance"

    def compute_flat_rewards(self, mols: list[RDMol], batch_idx: list[int]) -> tuple[FlatRewards, Tensor]:
        is_valid = [self.constraint(mol) for mol in mols]
        is_valid_t = torch.as_tensor(is_valid, dtype=torch.bool)
        valid_mols = [mol for mol, flag in zip(mols, is_valid, strict=True) if flag]
        vina_scores = self.run_vina(valid_mols)
        self.update_best_molecules(valid_mols, vina_scores.tolist())
        fr = vina_scores.neg().reshape(-1, 1)
        return FlatRewards(fr), is_valid_t

    def update_best_molecules(self, mols: list[RDMol], scores: list[float]):
        best_smi = [smi for score, smi in self.best_molecules]
        score_smiles = [(score, Chem.MolToSmiles(mol)) for score, mol in zip(scores, mols, strict=True)]
        self.last_molecules = score_smiles
        score_smiles = [(score, smi) for score, smi in score_smiles if smi not in best_smi]
        self.best_molecules = sorted(self.best_molecules + score_smiles, reverse=False)[:1000]

    def run_vina(self, mols: list[RDMol]) -> Tensor:
        out_dir = self.save_dir / f"oracle{self.oracle_idx}"
        vina_score = unidock_scores(mols, self.protein_path, self.center, search_mode=self.search_mode, out_dir=out_dir)
        self.oracle_idx += 1
        return torch.as_tensor(vina_score)

    def constraint(self, mol: RDMol) -> bool:
        if rdMolDescriptors.CalcExactMolWt(mol) > 500:
            return False
        if Lipinski.NumHDonors(mol) > 5:
            return False
        if Lipinski.NumHAcceptors(mol) > 10:
            return False
        if Lipinski.NumRotatableBonds(mol) > 5:
            return False
        return True


class Trainer2D(SynthesisTrainer):
    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.validate_every = 0
        cfg.algo.max_len = 2

    def setup_model(self):
        self.model = gfn.SynthesisGFN_DP(self.ctx, self.cfg)

    def setup_task(self):
        self.task: UniDockLipinskiTask = UniDockLipinskiTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        if len(self.task.last_molecules) > 0:
            info["batch_vina"] = np.mean([score for score, _ in self.task.best_molecules])
        if len(self.task.best_molecules) > 0:
            info["top1000_n"] = len(self.task.best_molecules)
            info["top1000_vina"] = np.mean([score for score, _ in self.task.best_molecules])
            info["top100_vina"] = np.mean([score for score, _ in self.task.best_molecules[:100]])
        super().log(info, index, key)


class Trainer3D(Trainer2D):
    def setup_env_context(self):
        self.ctx = Synthesis3DEnvContext(
            self.env,
            self.cfg.task.docking.protein_path,
            self.task.protein_path,
            self.cfg.task.docking.center,
            num_cond_dim=self.task.num_cond_dim,
            fp_radius_building_block=self.cfg.model.fp_radius_building_block,
            fp_nbits_building_block=self.cfg.model.fp_nbits_building_block,
        )
