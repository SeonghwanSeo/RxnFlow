import torch
import torch.nn as nn
from rdkit.Chem import QED

from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from rxnflow.config import Config
from rxnflow.base import RxnFlowTrainer, mogfn_trainer
from rxnflow.utils.chem_metrics import mol2qed, mol2sascore
from rxnflow.tasks.unidock import UniDockTask, UniDockTrainer


aux_tasks = {"qed": mol2qed, "sa": mol2sascore}


class UniDockMOOTask(UniDockTask):
    """Sets up a task where the reward is computed using a UniDock, QED."""

    is_moo = True

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        assert set(self.objectives) <= {"docking", "qed", "sa"}

    def compute_rewards(self, objs: list[RDMol]) -> Tensor:
        out_dir = self.save_dir / f"oracle{self.oracle_idx}"
        fr: Tensor
        flat_r: list[Tensor] = []
        self.avg_reward_info = []
        for prop in self.objectives:
            if prop == "docking":
                docking_scores = self.run_docking(objs, out_dir)
                self.update_storage(objs, docking_scores.tolist())
                fr = docking_scores * -0.1
            else:
                fr = aux_tasks[prop](objs)
            flat_r.append(fr)
            self.avg_reward_info.append((prop, fr.mean().item()))
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(objs)
        return flat_rewards

    def update_storage(self, objs: list[RDMol], scores: list[float]):

        def _filter(obj: RDMol) -> bool:
            """Check the object passes exact property filter"""
            return QED.qed(obj) > 0.5

        pass_idcs = [i for i, obj in enumerate(objs) if _filter(obj)]
        pass_objs = [objs[i] for i in pass_idcs]
        pass_scores = [scores[i] for i in pass_idcs]
        super().update_storage(pass_objs, pass_scores)


class UniDockMOO_PretrainTask(UniDockMOOTask):
    """Sets up a pretraining task where the reward is computed using a UniDock, QED."""

    def compute_rewards(self, objs: list[RDMol]) -> Tensor:
        fr: Tensor
        fr_dict: dict[str, Tensor] = {}
        self.avg_reward_info = []
        for obj in self.objectives:
            if obj == "docking":
                continue
            else:
                fr = aux_tasks[obj](objs)
            fr_dict[obj] = fr
            self.avg_reward_info.append((obj, fr.mean().item()))
        avg_fr = torch.stack(list(fr_dict.values()), -1).sum(-1) / (len(self.objectives) - 1)
        fr_dict["docking"] = avg_fr  # insert dummy values
        flat_r = [fr_dict[obj] for obj in self.objectives]
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(objs)
        return flat_rewards


@mogfn_trainer
class UniDockMOOTrainer(UniDockTrainer):
    task: UniDockMOOTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.validate_every = 0
        base.task.moo.objectives = ["docking", "qed"]
        base.num_training_steps = 1000

        base.cond.temperature.dist_params = [16, 64]
        base.replay.use = True
        base.replay.capacity = 6_400
        base.replay.warmup = 128
        base.cond.weighted_prefs.preference_type = "dirichlet"
        base.cond.focus_region.focus_type = None
        base.algo.train_random_action_prob = 0.02

    def setup_task(self):
        self.task = UniDockMOOTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


@mogfn_trainer
class UniDockMOO_Pretrainer(RxnFlowTrainer):
    task: UniDockMOO_PretrainTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.desc = "Vina-QED optimization with UniDock"
        base.validate_every = 0
        base.task.moo.objectives = ["docking", "qed"]
        base.cond.weighted_prefs.preference_type = "dirichlet"
        base.cond.focus_region.focus_type = None

        base.cond.temperature.dist_params = [16, 64]
        base.algo.train_random_action_prob = 0.1

    def setup_task(self):
        self.task = UniDockMOO_PretrainTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        for obj, v in self.task.avg_reward_info:
            info[f"sampled_{obj}_avg"] = v
        super().log(info, index, key)
