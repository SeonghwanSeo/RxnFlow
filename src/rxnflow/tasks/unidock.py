from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Lipinski, Crippen

from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet import ObjectProperties

from rxnflow.config import Config, init_empty
from rxnflow.base import BaseTask, RxnFlowTrainer
from rxnflow.utils.chem_metrics import mol2vina


class UniDockTask(BaseTask):
    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        self.protein_path: Path = Path(cfg.task.docking.protein_path)
        self.center: tuple[float, float, float] = cfg.task.docking.center
        self.size: tuple[float, float, float] = cfg.task.docking.size
        self.threshold: float = cfg.task.docking.threshold
        self.filter: str = cfg.task.constraint.rule
        assert self.filter in ["null", "lipinski", "veber"]

        self.last_molecules: list[tuple[float, str]] = []
        self.best_molecules: list[tuple[float, str]] = []
        self.save_dir: Path = Path(cfg.log_dir) / "unidock"
        self.search_mode: str = "balance"
        self.oracle_idx = 0

    def compute_obj_properties(self, objs: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        is_valid = [self.constraint(obj) for obj in objs]
        is_valid_t = torch.tensor(is_valid, dtype=torch.bool)
        valid_objs = [obj for flag, obj in zip(is_valid, objs, strict=True) if flag]
        if len(valid_objs) > 0:
            docking_scores = self.run_docking(valid_objs)
            self.update_storage(valid_objs, docking_scores.tolist())
            fr = self.convert_docking_score(docking_scores).reshape(-1, 1)
        else:
            fr = torch.zeros((0, 1), dtype=torch.float)
        return ObjectProperties(fr), is_valid_t

    def constraint(self, mol: RDMol) -> bool:
        if self.filter == "null":
            pass
        elif self.filter in ("lipinski", "veber"):
            if rdMolDescriptors.CalcExactMolWt(mol) > 500:
                return False
            if Lipinski.NumHDonors(mol) > 5:
                return False
            if Lipinski.NumHAcceptors(mol) > 5:
                return False
            if Crippen.MolLogP(mol) > 5:
                return False
            if self.filter == "veber":
                if rdMolDescriptors.CalcTPSA(mol) > 140:
                    return False
                if Lipinski.NumRotatableBonds(mol) > 10:
                    return False
        else:
            raise ValueError(self.filter)
        return True

    def convert_docking_score(self, scores: torch.Tensor):
        return self.threshold - scores

    def run_docking(self, mols: list[RDMol]) -> Tensor:
        out_dir = self.save_dir / f"oracle{self.oracle_idx}"
        docking_score = mol2vina(mols, self.protein_path, self.center, self.size, self.search_mode, out_dir)
        self.oracle_idx += 1
        return docking_score

    def update_storage(self, mols: list[RDMol], scores: list[float]):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        self.last_molecules = [(score, smi) for score, smi in zip(scores, smiles_list, strict=True)]

        best_smi = set(smi for _, smi in self.best_molecules)
        score_smiles = [(score, smi) for score, smi in self.last_molecules if smi not in best_smi]
        self.best_molecules = sorted(self.best_molecules + score_smiles, reverse=False)[:1000]


class UniDockTrainer(RxnFlowTrainer):
    task: UniDockTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.validate_every = 0
        base.num_training_steps = 1000

        # NOTE: Different to paper
        base.cond.temperature.dist_params = [16, 64]
        base.replay.use = True
        base.replay.capacity = 6_400
        base.replay.warmup = 128
        base.algo.train_random_action_prob = 0.05

    def setup_task(self):
        self.task = UniDockTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        if len(self.task.last_molecules) > 0:
            info["sample_docking_avg"] = np.mean([score for score, _ in self.task.last_molecules])
        if len(self.task.best_molecules) > 0:
            info["top100_n"] = len(self.task.best_molecules)
            info["top100_docking"] = np.mean([score for score, _ in self.task.best_molecules])
        super().log(info, index, key)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.num_training_steps = 100
    config.log_dir = "./logs/debug-unidock/"
    config.env_dir = "./data/envs/enamine_all"
    config.overwrite_existing_exp = True
    config.algo.action_subsampling.sampling_ratio = 0.1
    config.task.constraint.rule = "lipinski"

    config.task.docking.protein_path = "./data/experiments/LIT-PCBA/ADRB2.pdb"
    config.task.docking.center = (-1.96, -12.27, -48.98)

    trial = UniDockTrainer(config)
    trial.run()
