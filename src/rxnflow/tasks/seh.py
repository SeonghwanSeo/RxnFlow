from collections.abc import Callable

import torch
import torch_geometric.data as gd
from rdkit.Chem import Mol as RDMol
from torch import Tensor, nn

from gflownet import ObjectProperties
from gflownet.models import bengio2021flow
from gflownet.utils.misc import get_worker_device
from rxnflow.base import BaseTask, RxnFlowTrainer
from rxnflow.config import Config, init_empty


class SEHTask(BaseTask):
    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg)
        self._wrap_model: Callable[[nn.Module], nn.Module] = wrap_model
        self.models: dict[str, nn.Module] = self._load_task_models()

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model.to(get_worker_device())
        model = self._wrap_model(model)
        return {"seh": model}

    def compute_obj_properties(self, mols: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        graphs: list[gd.Data] = [bengio2021flow.mol2graph(i) for i in mols]
        assert len(graphs) == len(mols)
        is_valid = [i is not None for i in graphs]
        is_valid_t = torch.tensor(is_valid, dtype=torch.bool)
        if not is_valid_t.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid_t
        else:
            preds = self.calc_seh_reward(graphs).reshape((-1, 1))
            assert len(preds) == is_valid_t.sum()
            return ObjectProperties(preds), is_valid_t

    def calc_seh_reward(self, graphs: list[gd.Data]) -> Tensor:
        device = self.models["seh"].device if hasattr(self.models["seh"], "device") else get_worker_device()
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None]).to(device)
        preds = self.models["seh"](batch).reshape((-1,)).data.cpu() / 8
        preds[preds.isnan()] = 0
        return preds.clip(1e-4, 100).reshape((-1,))


class SEHTrainer(RxnFlowTrainer):
    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [0, 64.0]

    def setup_task(self):
        self.task = SEHTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    import datetime

    config = init_empty(Config())
    config.log_dir = f"./logs/debug/rxnflow-seh-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config.env_dir = "./data/envs/stock"

    config.print_every = 10
    config.num_training_steps = 10000
    config.num_workers_retrosynthesis = 8

    config.algo.action_subsampling.sampling_ratio = 0.1

    trial = SEHTrainer(config)
    try:
        trial.run()
    except Exception as e:
        print("terminate trainer")
        trial.terminate()
        raise e
