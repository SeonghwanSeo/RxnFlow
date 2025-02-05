import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import QED

from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet import ObjectProperties

from rxnflow.config import Config
from rxnflow.base import BaseTask, RxnFlowTrainer
from rxnflow.tasks.unidock import UniDockTask, UniDockTrainer
from rxnflow.tasks.utils.chem_metrics import mol2qed


aux_tasks = {"qed": mol2qed}


class UniDockMOO_PretrainTask(BaseTask):
    """Sets up a task where the reward is computed using a UniDock, QED."""

    is_moo = True

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        assert set(self.objectives) <= {"docking", "qed"}

    def compute_obj_properties(self, objs: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        is_valid_t = torch.ones(len(objs), dtype=torch.bool)
        fr: Tensor
        fr_dict: dict[str, Tensor] = {}
        fr_sum: Tensor = torch.zeros(len(objs))
        self.avg_reward_info = []
        for obj in self.objectives:
            if obj == "docking":
                continue
            else:
                fr = aux_tasks[obj](objs)
            fr_dict[obj] = fr
            fr_sum = fr_sum + fr
            self.avg_reward_info.append((obj, fr.mean().item()))
        fr_dict["docking"] = fr_sum / (len(self.objectives) - 1)
        flat_r = [fr_dict[obj] for obj in self.objectives]
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(objs)
        return ObjectProperties(flat_rewards), is_valid_t


class UniDockMOOTask(UniDockTask):
    """Sets up a task where the reward is computed using a UniDock, QED."""

    is_moo = True

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        assert set(self.objectives) <= {"docking", "qed"}

    def compute_obj_properties(self, objs: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        is_valid = [self.constraint(obj) for obj in objs]
        is_valid_t = torch.tensor(is_valid, dtype=torch.bool)
        valid_objs = [obj for flag, obj in zip(is_valid, objs, strict=True) if flag]

        fr: Tensor
        flat_r: list[Tensor] = []
        self.avg_reward_info = []
        for prop in self.objectives:
            if prop == "docking":
                docking_scores = self.run_docking(valid_objs)
                self.update_storage(valid_objs, docking_scores.tolist())
                fr = docking_scores * -0.1
            else:
                fr = aux_tasks[prop](valid_objs)
            flat_r.append(fr)
            self.avg_reward_info.append((prop, fr.mean().item()))
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(valid_objs)
        return ObjectProperties(flat_rewards), is_valid_t

    def update_storage(self, mols: list[RDMol], scores: list[float]):
        self.last_molecules = [
            (score, Chem.MolToSmiles(mol)) for score, mol in zip(scores, mols, strict=True) if QED.qed(mol) > 0.5
        ]

        best_smi = set(smi for _, smi in self.best_molecules)
        score_smiles = [(score, smi) for score, smi in self.last_molecules if smi not in best_smi]
        self.best_molecules = sorted(self.best_molecules + score_smiles, reverse=False)[:1000]


class UniDockMOOTrainer(UniDockTrainer):
    task: UniDockMOOTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.validate_every = 0
        base.task.moo.objectives = ["docking", "qed"]
        base.task.constraint.rule = None
        base.num_training_steps = 1000

        # NOTE: Different to paper
        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [1, 64]
        base.algo.train_random_action_prob = 0.01
        base.replay.use = True
        base.replay.capacity = 6_400
        base.replay.warmup = 128

    def setup_task(self):
        self.task = UniDockMOOTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        for obj, v in self.task.avg_reward_info:
            info[f"sampled_{obj}_avg"] = v
        info["topn"] = len(self.task.best_molecules)
        for n in (10, 100, 1000):
            info[f"top{n}_docking"] = np.mean([score for score, _ in self.task.best_molecules[:n]])
        super().log(info, index, key)


class UniDockMOO_Pretrainer(RxnFlowTrainer):
    task: UniDockMOO_PretrainTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.desc = "Vina-QED optimization with UniDock"
        base.validate_every = 0
        base.task.moo.objectives = ["docking", "qed"]
        base.cond.weighted_prefs.preference_type = "dirichlet"
        base.cond.focus_region.focus_type = None

        base.algo.sampling_tau = 0.0
        base.cond.temperature.dist_params = [16, 64]  # Different to Paper!
        base.algo.train_random_action_prob = 0.1

    def setup_task(self):
        self.task = UniDockMOO_PretrainTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        for obj, v in self.task.avg_reward_info:
            info[f"sampled_{obj}_avg"] = v
        super().log(info, index, key)
