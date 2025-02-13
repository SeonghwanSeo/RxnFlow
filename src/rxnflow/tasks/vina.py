from pathlib import Path
from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen
from rdkit.Chem import Mol as RDMol

from gflownet import ObjectProperties

from rxnflow.config import Config, init_empty
from rxnflow.base import BaseTask, RxnFlowTrainer, RxnFlowSampler
from rxnflow.utils.misc import create_logger
from rxnflow.tasks.utils.unidock import VinaReward


class VinaTask(BaseTask):
    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        self.oracle_idx = 0
        self.filter = cfg.task.constraint.rule

        self.vina = VinaReward(
            cfg.task.docking.protein_path,
            cfg.task.docking.center,
            cfg.task.docking.ref_ligand_path,
            cfg.task.docking.size,
            search_mode="balance",  # fast, balance, detail
            num_workers=4,
        )

        self.topn_vina: OrderedDict[str, float] = OrderedDict()
        self.batch_vina: list[float] = []

        self.save_dir: Path = Path(cfg.log_dir) / "docking"
        self.save_dir.mkdir()

    def compute_obj_properties(self, objs: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        is_valid_t = torch.tensor([self.constraint(obj) for obj in objs], dtype=torch.bool)
        valid_objs = [obj for flag, obj in zip(is_valid_t, objs, strict=True) if flag]
        if len(valid_objs) > 0:
            docking_scores = self.mol2vina(valid_objs)
            fr = docking_scores * -0.1
        else:
            fr = torch.zeros((0,), dtype=torch.float)
        self.oracle_idx += 1
        return ObjectProperties(fr.reshape(-1, 1)), is_valid_t

    def constraint(self, mol: RDMol) -> bool:
        if self.filter is None:
            pass
        elif self.filter in ("lipinski", "veber"):
            if rdMolDescriptors.CalcExactMolWt(mol) > 500:
                return False
            if rdMolDescriptors.CalcNumHBD(mol) > 5:
                return False
            if rdMolDescriptors.CalcNumHBA(mol) > 10:
                return False
            if Crippen.MolLogP(mol) > 5:
                return False
            if self.filter == "veber":
                if rdMolDescriptors.CalcTPSA(mol) > 140:
                    return False
                if rdMolDescriptors.CalcNumRotatableBonds(mol) > 10:
                    return False
        else:
            raise ValueError(self.filter)
        return True

    def mol2vina(self, mols: list[RDMol]) -> torch.Tensor:
        out_path = self.save_dir / f"oracle{self.oracle_idx}.sdf"
        vina_scores = self.vina.run_mols(mols, out_path)
        self.update_storage(mols, vina_scores)
        return torch.tensor(vina_scores, dtype=torch.float32)

    def update_storage(self, objs: list[RDMol], vina_scores: list[float]):
        """update vina metric"""
        self.batch_vina = vina_scores
        smiles_list = [Chem.MolToSmiles(obj) for obj in objs]
        self.topn_vina.update(zip(smiles_list, vina_scores, strict=True))
        topn = sorted(list(self.topn_vina.items()), key=lambda v: v[1])[:1000]
        self.topn_vina = OrderedDict(topn)


class VinaTrainer(RxnFlowTrainer):
    task: VinaTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.print_every = 1
        base.validate_every = 0
        base.num_training_steps = 1000
        base.task.constraint.rule = "lipinski"

        base.algo.max_len = 3
        base.algo.train_random_action_prob = 0.05
        base.algo.action_subsampling.sampling_ratio = 0.01

        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [0.0, 64.0]
        base.replay.use = True
        base.replay.capacity = 6_400
        base.replay.warmup = 0

    def setup_task(self):
        self.task = VinaTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        self.add_extra_info(info)
        super().log(info, index, key)

    def add_extra_info(self, info):
        if self.task.filter is not None:
            info["pass_constraint"] = len(self.task.batch_vina) / self.cfg.algo.num_from_policy
        if len(self.task.batch_vina) > 0:
            info["sampled_vina_avg"] = np.mean(self.task.batch_vina)
        best_vinas = list(self.task.topn_vina.values())
        for topn in [10, 100, 1000]:
            if len(best_vinas) > topn:
                info[f"top{topn}_vina"] = np.mean(best_vinas[:topn])


# NOTE: Sampling with pre-trained GFlowNet
class VinaSampler(RxnFlowSampler):
    def setup_task(self):
        self.task: VinaTask = VinaTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.num_training_steps = 100
    config.log_dir = "./logs/debug-vina/"
    config.env_dir = "./data/envs/stock"
    config.overwrite_existing_exp = True
    config.algo.max_len = 2
    config.task.constraint.rule = "lipinski"

    config.task.docking.protein_path = "./data/examples/6oim_protein.pdb"
    config.task.docking.center = (1.872, -8.260, -1.361)

    # to prevent redundant logging due to unidock
    logger = create_logger()
    trial = VinaTrainer(config)
    trial.run(logger)
