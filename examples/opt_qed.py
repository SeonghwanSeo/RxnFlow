import torch
from torch import Tensor
from rdkit.Chem import QED, Mol as RDMol

from gflownet import ObjectProperties
from rxnflow.config import Config, init_empty
from rxnflow.base import RxnFlowTrainer, BaseTask


class QEDTask(BaseTask):
    def compute_obj_properties(self, objs: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        fr = torch.tensor([QED.qed(obj) for obj in objs], dtype=torch.float32)
        fr = fr.reshape(-1, 1)
        is_valid_t = torch.ones((len(objs),), dtype=torch.bool)
        return ObjectProperties(fr), is_valid_t


class QEDTrainer(RxnFlowTrainer):  # For online training
    def setup_task(self):
        self.task = QEDTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


if __name__ == "__main__":
    config = init_empty(Config())
    config.log_dir = "./logs/example-qed"
    config.overwrite_existing_exp = True
    config.checkpoint_every = 1000
    config.store_all_checkpoints = True
    config.print_every = 1
    config.num_workers_retrosynthesis = 8

    config.env_dir = "./data/envs/stock/"
    config.num_training_steps = 10000
    config.algo.action_subsampling.sampling_ratio = 0.01

    config.cond.temperature.dist_params = [0, 64]
    config.algo.train_random_action_prob = 0.1

    # use replay buffer
    config.replay.use = False
    config.replay.capacity = 10_000
    config.replay.warmup = 1_000

    trainer = QEDTrainer(config)
    trainer.run()
