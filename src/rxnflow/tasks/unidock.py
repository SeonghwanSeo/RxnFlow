from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Lipinski, Crippen

from collections import OrderedDict
from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from rxnflow.config import Config, init_empty
from rxnflow.base import BaseTask, RxnFlowTrainer, RxnFlowSampler
from rxnflow.utils.unidock import run_docking


class UniDockTask(BaseTask):
    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        self.protein_path: Path = Path(cfg.task.docking.protein_path)
        self.center: tuple[float, float, float] = cfg.task.docking.center
        self.size: tuple[float, float, float] = cfg.task.docking.size
        self.filter: str | None = cfg.task.constraint.rule
        self.ff_optimization: None | str = None  # None, UFF, MMFF

        self.search_mode: str = "fast"  # fast, balance, detail
        assert self.filter in [None, "lipinski", "veber"]

        self.save_dir: Path = Path(cfg.log_dir) / "docking"
        self.save_dir.mkdir()

        self.topn_vina: OrderedDict[str, float] = OrderedDict()
        self.batch_vina: list[float]

    def compute_rewards(self, objs: list[RDMol]) -> Tensor:
        docking_scores = self.run_docking(objs)
        fr = docking_scores.neg().clip(1e-5)
        return fr.reshape(-1, 1)

    def filter_object(self, obj: RDMol) -> bool:
        if self.filter is None:
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

    def run_docking(self, objs: list[RDMol]) -> Tensor:
        out_path = self.save_dir / f"oracle{self.oracle_idx}.sdf"
        vina_score = run_docking(
            objs,
            self.protein_path,
            out_path,
            self.center,
            self.size,
            seed=1,
            search_mode=self.search_mode,
            ff_optimization=self.ff_optimization,
        )
        self.update_storage(objs, vina_score)
        return torch.tensor(vina_score, dtype=torch.float).clip(max=0.0)

    def update_storage(self, objs: list[RDMol], scores: list[float]):
        self.batch_vina = scores
        smiles_list = [Chem.MolToSmiles(obj) for obj in objs]
        self.topn_vina.update(zip(smiles_list, scores, strict=True))
        if len(self.topn_vina) > 2000:
            topn = sorted(list(self.topn_vina.items()), key=lambda v: v[1])[:2000]
            self.topn_vina = OrderedDict(topn)


class UniDockTrainer(RxnFlowTrainer):
    task: UniDockTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.num_training_steps = 1000
        base.task.constraint.rule = None

        base.cond.temperature.sample_dist = "constant"
        base.cond.temperature.dist_params = [32.0]

        base.replay.use = True
        base.replay.capacity = 6_400
        base.replay.warmup = 128
        base.algo.train_random_action_prob = 0.05

    def setup_task(self):
        self.task = UniDockTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        self.add_extra_info(info)
        super().log(info, index, key)

    def add_extra_info(self, info):
        if len(self.task.batch_vina) > 0:
            info["sample_docking_avg"] = np.mean(self.task.batch_vina)
        best_vinas = list(self.task.topn_vina.values())
        for topn in [10, 100, 1000]:
            if len(best_vinas) > topn:
                info[f"top{topn}_docking"] = np.mean(best_vinas[:topn])


# NOTE: Sampling with pre-trained GFlowNet
class UniDockSampler(RxnFlowSampler):
    def setup_task(self):
        self.task: UniDockTask = UniDockTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = "./logs/debug-unidock/"
    config.env_dir = "./data/envs/stock"
    config.task.constraint.rule = "veber"
    config.overwrite_existing_exp = True
    config.algo.action_subsampling.sampling_ratio = 0.01

    config.task.docking.protein_path = "./data/examples/6oim_protein.pdb"
    config.task.docking.center = (1.872, -8.260, -1.361)

    trial = UniDockTrainer(config)
    trial.run()
