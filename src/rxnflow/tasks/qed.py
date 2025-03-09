import torch
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet import ObjectProperties
from rxnflow.base import BaseTask, RxnFlowSampler, RxnFlowTrainer
from rxnflow.config import Config, init_empty
from rxnflow.tasks.utils.chem_metrics import mol2qed


class QEDTask(BaseTask):
    def compute_obj_properties(self, mols: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        fr = mol2qed(mols).reshape(-1, 1)
        is_valid_t = torch.ones((len(mols),), dtype=torch.bool)
        return ObjectProperties(fr), is_valid_t


class QEDTrainer(RxnFlowTrainer):  # For online training
    def setup_task(self):
        self.task = QEDTask(self.cfg)


class QEDSampler(RxnFlowSampler):  # Sampling with pre-trained GFlowNet
    def setup_task(self):
        self.task = QEDTask(self.cfg)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    import datetime

    config = init_empty(Config())
    config.log_dir = f"./logs/debug/rxnflow-qed-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config.env_dir = "./data/envs/stock"

    config.print_every = 1
    config.num_training_steps = 100
    config.num_workers_retrosynthesis = 4

    config.algo.action_subsampling.sampling_ratio = 0.1

    trial = QEDTrainer(config)
    try:
        trial.run()
    except Exception as e:
        print("terminate trainer")
        trial.terminate()
        raise e
