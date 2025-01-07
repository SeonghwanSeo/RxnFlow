import torch
import torch.nn as nn
from rdkit.Chem import QED

from collections import OrderedDict
from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from rxnflow.config import Config, init_empty
from rxnflow.base import mogfn_trainer
from rxnflow.utils.chem_metrics import mol2qed, mol2sascore
from rxnflow.tasks.unidock import UniDockTask, UniDockTrainer


aux_tasks = {"qed": mol2qed, "sa": mol2sascore}


class UniDockMOOTask(UniDockTask):
    """Sets up a task where the reward is computed using a UniDock, QED."""

    is_moo = True

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        assert set(self.objectives) <= {"vina", "qed", "sa"}

    def compute_rewards(self, objs: list[RDMol]) -> Tensor:
        fr: Tensor
        flat_r: list[Tensor] = []
        self.avg_reward_info = OrderedDict()
        for prop in self.objectives:
            if prop == "vina":
                fr = self.run_docking(objs) * -0.1
            else:
                fr = aux_tasks[prop](objs)
            flat_r.append(fr)
            self.avg_reward_info[prop] = fr.mean().item()
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(objs)
        return flat_rewards

    def update_storage(self, objs: list[RDMol], scores: list[float]):
        def _filter(obj: RDMol) -> bool:
            """Check the object passes a property filter"""
            return QED.qed(obj) > 0.5

        pass_idcs = [i for i, obj in enumerate(objs) if _filter(obj)]
        pass_objs = [objs[i] for i in pass_idcs]
        pass_scores = [scores[i] for i in pass_idcs]
        super().update_storage(pass_objs, pass_scores)


@mogfn_trainer
class UniDockMOOTrainer(UniDockTrainer):
    task: UniDockMOOTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.num_training_steps = 1000
        base.task.constraint.rule = None
        base.task.moo.objectives = ["vina", "qed"]

        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [0, 64]
        base.algo.train_random_action_prob = 0.05
        base.cond.weighted_prefs.preference_type = "dirichlet"
        base.cond.focus_region.focus_type = None

    def setup_task(self):
        self.task = UniDockMOOTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def add_extra_info(self, info):
        for prop, fr in self.task.avg_reward_info.items():
            info[f"sample_r_{prop}_avg"] = fr
        super().add_extra_info(info)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = "./logs/debug-unidock/"
    config.env_dir = "./data/envs/stock"
    config.overwrite_existing_exp = True
    config.algo.action_subsampling.sampling_ratio = 0.01

    config.task.docking.protein_path = "./data/examples/6oim_protein.pdb"
    config.task.docking.center = (1.872, -8.260, -1.361)

    trial = UniDockMOOTrainer(config)
    trial.run()
