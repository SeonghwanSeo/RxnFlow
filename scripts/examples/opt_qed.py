import torch
from rdkit.Chem import QED
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet import ObjectProperties
from rxnflow.base import BaseTask, RxnFlowTrainer
from rxnflow.config import Config, init_empty


class QEDTask(BaseTask):
    def compute_obj_properties(self, mols: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        fr = torch.tensor([QED.qed(obj) for obj in mols], dtype=torch.float32)
        fr = fr.reshape(-1, 1)
        is_valid_t = torch.ones((len(mols),), dtype=torch.bool)
        return ObjectProperties(fr), is_valid_t


class QEDTrainer(RxnFlowTrainer):  # For online training
    def setup_task(self):
        self.task = QEDTask(self.cfg)


if __name__ == "__main__":
    config = init_empty(Config())
    config.log_dir = "./logs/example/qed"
    config.env_dir = "./data/envs/catalog/"
    config.overwrite_existing_exp = True
    config.num_training_steps = 10000
    config.checkpoint_every = 1000
    config.store_all_checkpoints = True
    config.print_every = 1
    config.num_workers_retrosynthesis = 4

    config.algo.action_subsampling.sampling_ratio = 0.02

    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0, 64]
    config.algo.train_random_action_prob = 0.1

    # use replay buffer
    config.replay.use = False
    config.replay.capacity = 10_000
    config.replay.warmup = 1_000

    trainer = QEDTrainer(config)
    trainer.run()
