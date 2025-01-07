from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Lipinski, Crippen

from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from rxnflow.config import Config, init_empty
from rxnflow.base import BaseTask, RxnFlowTrainer, RxnFlowSampler
from rxnflow.utils.unidock import unidock_scores


class UniDockTask(BaseTask):
    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        self.protein_path: Path = Path(cfg.task.docking.protein_path)
        self.center: tuple[float, float, float] = cfg.task.docking.center
        self.size: tuple[float, float, float] = cfg.task.docking.size
        self.threshold: float = cfg.task.docking.threshold
        self.filter: str = cfg.task.constraint.rule
        self.ff_optimization: None | str = None  # None, UFF, MMFF
        self.search_mode: str = "balance"  # fast, balance, detail
        assert self.filter in ["null", "lipinski", "veber"]

        self.save_dir: Path = Path(cfg.log_dir) / "docking"
        self.save_dir.mkdir()
        self.batch_scores: list[float] = []
        self.best_scores: list[float] = []
        self.batch_novelty: float = 0.0
        self.history: set[str] = set()

    def compute_rewards(self, objs: list[RDMol]) -> Tensor:
        out_dir = self.save_dir / f"oracle{self.oracle_idx}"
        docking_scores = self.run_docking(objs, out_dir)
        fr = self.convert_docking_score(docking_scores)
        self.update_storage(objs, docking_scores.tolist())
        return fr.reshape(-1, 1)

    def convert_docking_score(self, scores: torch.Tensor):
        return self.threshold - scores

    def filter_object(self, obj: RDMol) -> bool:
        if self.filter == "null":
            pass
        elif self.filter in ("lipinski", "veber"):
            if rdMolDescriptors.CalcExactMolWt(obj) > 500:
                return False
            if Lipinski.NumHDonors(obj) > 5:
                return False
            if Lipinski.NumHAcceptors(obj) > 10:
                return False
            if Crippen.MolLogP(obj) > 5:
                return False
            if self.filter == "veber":
                if rdMolDescriptors.CalcTPSA(obj) > 140:
                    return False
                if Lipinski.NumRotatableBonds(obj) > 10:
                    return False
        else:
            raise ValueError(self.filter)
        return True

    def run_docking(self, mols: list[RDMol], out_dir: Path) -> Tensor:
        vina_score = unidock_scores(
            mols,
            self.protein_path,
            out_dir,
            self.center,
            self.size,
            seed=1,
            search_mode=self.search_mode,
            ff_optimization=self.ff_optimization,
        )
        return torch.tensor(vina_score, dtype=torch.float).clip(max=0.0)

    def update_storage(self, objs: list[RDMol], scores: list[float]):
        self.batch_scores = scores
        smiles_list = [Chem.MolToSmiles(obj) for obj in objs]

        score_dict = {smi: v for smi, v in zip(smiles_list, scores, strict=True)}
        nov_uniq_smi = list(set(smiles_list).difference(self.history))
        self.history.update(nov_uniq_smi)

        nov_uniq_scores = [score_dict[smi] for smi in nov_uniq_smi]
        self.best_scores = self.best_scores + nov_uniq_scores
        self.best_scores.sort()
        self.best_scores = self.best_scores[:1000]


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
        base.algo.train_random_action_prob = 0.1

    def setup_task(self):
        self.task = UniDockTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        self.add_extra_info(info)
        super().log(info, index, key)

    def add_extra_info(self, info):
        if len(self.task.batch_scores) > 0:
            info["sample_docking_avg"] = np.mean(self.task.batch_scores)
        info["top_n"] = len(self.task.best_scores)
        for topn in [10, 50, 100, 500, 1000]:
            info[f"top{topn}_docking"] = np.mean(self.task.best_scores[:topn])


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
    config.algo.action_subsampling.sampling_ratio = 0.1

    config.task.docking.protein_path = "./data/examples/6oim_protein.pdb"
    config.task.docking.center = (1.872, -8.260, -1.361)

    trial = UniDockTrainer(config)
    trial.run()
