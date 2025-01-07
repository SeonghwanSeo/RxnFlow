import torch_geometric.data as gd

from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet.utils.misc import get_worker_device
from rxnflow.config import Config, init_empty
from rxnflow.base import BaseTask, RxnFlowTrainer
from gflownet.models import bengio2021flow


class SEHTask(BaseTask):
    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model.to(get_worker_device())
        model = self._wrap_model(model)
        return {"seh": model}

    def compute_rewards(self, objs: list[RDMol]) -> Tensor:
        graphs = [bengio2021flow.mol2graph(i) for i in objs]
        preds = self.compute_reward_from_graph(graphs).reshape((-1, 1))
        assert len(preds) == len(objs)
        return preds

    def compute_reward_from_graph(self, graphs: list[gd.Data]) -> Tensor:
        device = self.models["seh"].device if hasattr(self.models["seh"], "device") else get_worker_device()
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None]).to(device)
        preds = self.models["seh"](batch).reshape((-1,)).data.cpu() / 8
        preds[preds.isnan()] = 0
        return preds.clip(1e-4, 100).reshape((-1,))


class SEHTrainer(RxnFlowTrainer):
    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.algo.sampling_tau = 0.95
        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [0, 64.0]

    def setup_task(self):
        self.task = SEHTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.log_dir = "./logs/debug-seh/"
    config.env_dir = "./data/envs/stock"
    config.overwrite_existing_exp = True

    config.print_every = 10
    config.num_training_steps = 10000
    config.num_workers_retrosynthesis = 4

    config.algo.max_len = 3
    config.algo.train_random_action_prob = 0.05
    config.algo.action_subsampling.sampling_ratio = 0.05

    trial = SEHTrainer(config)
    trial.run()
