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
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.models["seh"].device if hasattr(self.models["seh"], "device") else get_worker_device())
        preds = self.models["seh"](batch).reshape((-1,)).data.cpu() / 8
        preds[preds.isnan()] = 0
        return preds.clip(1e-4, 100).reshape((-1,))


class SEHTrainer(RxnFlowTrainer):
    def setup_task(self):
        self.task = SEHTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.num_training_steps = 100
    config.log_dir = "./logs/debug-synple/"
    config.env_dir = "./data/envs/real"
    config.overwrite_existing_exp = True
    config.algo.action_subsampling.sampling_ratio = 0.1

    trial = SEHTrainer(config)
    trial.run()
