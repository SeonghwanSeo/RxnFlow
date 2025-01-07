import torch_geometric.data as gd

import torch
from torch import nn

from collections.abc import Callable
from torch import Tensor
from rdkit.Chem import Mol as RDMol

from gflownet.utils.misc import get_worker_device
from gflownet.models import bengio2021flow
from rxnflow.config import Config, init_empty
from rxnflow.base import RxnFlowTrainer, mogfn_trainer
from rxnflow.tasks.seh import SEHTask
from rxnflow.utils.chem_metrics import mol2qed, mol2mw, mol2sascore


aux_tasks = {"qed": mol2qed, "mw": mol2mw, "sa": mol2sascore}


class SEHMOOTask(SEHTask):
    is_moo = True

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        assert set(self.objectives) <= {"seh", "qed", "mw", "sa"}

    def compute_rewards(self, objs: list[RDMol]) -> Tensor:
        flat_r: list[Tensor] = []
        for obj in self.objectives:
            if obj == "seh":
                graphs = [bengio2021flow.mol2graph(i) for i in objs]
                flat_r.append(super().compute_reward_from_graph(graphs))
            else:
                flat_r.append(aux_tasks[obj](objs))

        flat_rewards = torch.stack(flat_r, dim=1)
        return flat_rewards

    def compute_reward_from_graph(self, graphs: list[gd.Data]) -> Tensor:
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.models["seh"].device if hasattr(self.models["seh"], "device") else get_worker_device())
        preds = self.models["seh"](batch).reshape((-1,)).data.cpu() / 8
        preds[preds.isnan()] = 0
        return preds.clip(1e-4, 100).reshape((-1,))


@mogfn_trainer
class SEHMOOTrainer(RxnFlowTrainer):
    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.algo.sampling_tau = 0.95
        base.cond.temperature.sample_dist = "constant"
        base.cond.temperature.dist_params = [32.0]

    def setup_task(self):
        self.task = SEHMOOTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 10
    config.num_workers_retrosynthesis = 4
    config.num_training_steps = 10000
    config.log_dir = "./logs/debug-seh-qed/"
    config.env_dir = "./data/envs/stock"
    config.task.moo.objectives = ["seh", "qed"]
    config.overwrite_existing_exp = True
    config.algo.action_subsampling.sampling_ratio = 0.01

    trial = SEHMOOTrainer(config)
    trial.run()
