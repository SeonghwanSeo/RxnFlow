import torch
import math

from typing import List, Tuple
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet.config import Config, init_empty
from gflownet.misc.toy.algo import ToyTrajectoryBalance
from gflownet.tasks.synthesis_trainer import SynthesisTrainer
from gflownet.tasks.base_task import BaseTask
from gflownet.trainer import FlatRewards


class SizeTask(BaseTask):
    Z = 15.923479423492735

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        rewards = torch.tensor([mol.GetNumHeavyAtoms() for mol in mols], dtype=torch.float).unsqueeze(-1)
        rewards /= math.exp(self.Z - 15)  # normalize to Z = 15
        return FlatRewards(rewards), torch.ones((len(mols),), dtype=torch.bool)


class ToyTrainer(SynthesisTrainer):
    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.desc = "Toy Task"
        cfg.env_dir = "./data/experiments/toy/env_1000"
        cfg.algo.min_len = 1
        cfg.algo.max_len = 2
        cfg.algo.sampling_tau = 0
        cfg.algo.train_random_action_prob = 0.01
        cfg.algo.action_sampling.max_sampling_reactbi = 1000
        cfg.algo.action_sampling.min_sampling_reactbi = 1
        cfg.cond.temperature.sample_dist = "constant"
        cfg.cond.temperature.dist_params = [1]

        cfg.num_training_steps = 40000
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 4000
        cfg.opt.learning_rate = 1e-4
        cfg.opt.lr_decay = 2000

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        algo = ToyTrajectoryBalance
        self.algo = algo(self.env, self.ctx, self.rng, self.cfg)


class ToySizeTrainer(ToyTrainer):
    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.desc = "Toy: NumHeavyAtoms"

    def setup_task(self):
        self.task = SizeTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)


def main():
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.validate_every = 100
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True
    SAMPLING_RATIO = 1.0
    config.algo.action_sampling.num_mc_sampling = 1
    config.algo.action_sampling.num_sampling_add_first_reactant = int(1000 * SAMPLING_RATIO)
    config.algo.action_sampling.sampling_ratio_reactbi = SAMPLING_RATIO
    config.log_dir = "./logs/debug/"
    trial = ToySizeTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
