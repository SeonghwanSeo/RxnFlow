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
from rxnflow.base import BaseTask, RxnFlowTrainer, RxnFlowSampler
from rxnflow.tasks.utils.unidock import unidock_scores


class UniDockTask(BaseTask):
    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        self.protein_path: Path = Path(cfg.task.docking.protein_path)
        self.center: tuple[float, float, float] = cfg.task.docking.center
        self.size: tuple[float, float, float] = cfg.task.docking.size
        self.filter: str | None = cfg.task.constraint.rule
        self.ff_optimization: None | str = None  # None, UFF, MMFF
        self.search_mode: str = "balance"  # fast, balance, detail
        assert self.filter in [None, "lipinski", "veber"]

        self.last_molecules: list[tuple[float, str]] = []
        self.best_molecules: list[tuple[float, str]] = []
        self.save_dir: Path = Path(cfg.log_dir) / "docking"
        self.save_dir.mkdir()
        self.oracle_idx = 0

    def compute_obj_properties(self, objs: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        is_valid_t = torch.ones(len(objs), dtype=torch.bool)

        fr = torch.zeros(len(objs), dtype=torch.float)
        is_pass = [self.constraint(obj) for obj in objs]
        valid_objs = [obj for flag, obj in zip(is_pass, objs, strict=True) if flag]
        if len(valid_objs) > 0:
            docking_scores = self.run_docking(valid_objs)
            self.update_storage(valid_objs, docking_scores.tolist())
            fr[is_pass] = self.convert_docking_score(docking_scores)
        return ObjectProperties(fr.reshape(-1, 1)), is_valid_t

    def constraint(self, mol: RDMol) -> bool:
        if self.filter is None:
            pass
        elif self.filter in ("lipinski", "veber"):
            if rdMolDescriptors.CalcExactMolWt(mol) > 500:
                return False
            if Lipinski.NumHDonors(mol) > 5:
                return False
            if Lipinski.NumHAcceptors(mol) > 10:
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
        return -scores

    def run_docking(self, mols: list[RDMol]) -> Tensor:
        out_path = self.save_dir / f"oracle{self.oracle_idx}.sdf"
        vina_score = unidock_scores(
            mols,
            self.protein_path,
            self.center,
            out_path,
            self.size,
            seed=1,
            search_mode=self.search_mode,
            ff_optimization=self.ff_optimization,
        )
        self.oracle_idx += 1
        return torch.tensor(vina_score, dtype=torch.float).clip(max=0.0)

    def update_storage(self, mols: list[RDMol], scores: list[float]):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        self.last_molecules = [(score, smi) for score, smi in zip(scores, smiles_list, strict=True)]

        best_smi = set(smi for _, smi in self.best_molecules)
        score_smiles = [(score, smi) for score, smi in self.last_molecules if smi not in best_smi]
        self.best_molecules = self.best_molecules + score_smiles
        self.best_molecules.sort(key=lambda v: v[0])
        self.best_molecules = self.best_molecules[:1000]


class UniDockTrainer(RxnFlowTrainer):
    task: UniDockTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.print_every = 1
        base.validate_every = 0
        base.num_training_steps = 1000
        base.task.constraint.rule = "lipinski"

        base.cond.temperature.sample_dist = "constant"
        base.cond.temperature.dist_params = [32.0]
        base.replay.use = True
        base.replay.capacity = 6_400
        base.replay.warmup = 256
        base.algo.train_random_action_prob = 0.01

    def setup_task(self):
        self.task = UniDockTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        self.add_extra_info(info)
        super().log(info, index, key)

    def add_extra_info(self, info):
        if self.task.filter != "null":
            info["pass_constraint"] = len(self.task.last_molecules) / self.cfg.algo.num_from_policy
        if len(self.task.last_molecules) > 0:
            info["sample_docking_avg"] = np.mean([score for score, _ in self.task.last_molecules])
        info["topn"] = len(self.task.best_molecules)
        for n in (10, 100, 1000):
            info[f"top{n}_docking"] = np.mean([score for score, _ in self.task.best_molecules[:n]])


# NOTE: Sampling with pre-trained GFlowNet
class UniDockSampler(RxnFlowSampler):
    def setup_task(self):
        self.task: UniDockTask = UniDockTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.num_training_steps = 100
    config.log_dir = "./logs/debug-unidock/"
    config.env_dir = "./data/envs/real"
    config.overwrite_existing_exp = True
    config.algo.max_len = 2
    config.algo.action_subsampling.sampling_ratio = 0.01
    config.task.constraint.rule = "lipinski"

    config.task.docking.protein_path = "./data/examples/6oim_protein.pdb"
    config.task.docking.center = (1.872, -8.260, -1.361)

    trial = UniDockTrainer(config)
    trial.run()
