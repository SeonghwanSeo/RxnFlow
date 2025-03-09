from collections.abc import Callable

import torch
from rdkit.Chem import Mol as RDMol
from torch import Tensor, nn

from gflownet import ObjectProperties
from gflownet.models import bengio2021flow
from gflownet.tasks.seh_frag_moo import aux_tasks
from rxnflow.config import Config, init_empty
from rxnflow.tasks.seh import SEHTask, SEHTrainer


class SEHMOOTask(SEHTask):
    is_moo = True

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        assert set(self.objectives) <= {"seh", "qed", "sa", "mw"} and len(self.objectives) == len(set(self.objectives))

    def compute_obj_properties(self, mols: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        assert len(graphs) == len(mols)
        is_valid = [i is not None for i in graphs]
        is_valid_t = torch.tensor(is_valid, dtype=torch.bool)
        if not any(is_valid):
            return ObjectProperties(torch.zeros((0, len(self.objectives)))), is_valid_t
        else:
            flat_r: list[Tensor] = []
            for obj in self.objectives:
                if obj == "seh":
                    flat_r.append(self.calc_seh_reward(graphs))
                else:
                    flat_r.append(aux_tasks[obj](mols, is_valid))

            flat_rewards = torch.stack(flat_r, dim=1)
            assert flat_rewards.shape[0] == len(mols)
            return ObjectProperties(flat_rewards), is_valid_t


class SEHMOOTrainer(SEHTrainer):
    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.algo.sampling_tau = 0.95

    def setup_task(self):
        self.task = SEHMOOTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    import datetime

    config = init_empty(Config())
    config.log_dir = f"./logs/debug/rxnflow-sehmoo-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config.env_dir = "./data/envs/stock"
    config.task.moo.objectives = ["seh", "qed"]

    config.print_every = 10
    config.num_training_steps = 10000
    config.num_workers_retrosynthesis = 4

    config.algo.action_subsampling.sampling_ratio = 0.1

    trial = SEHMOOTrainer(config)
    trial.run()
