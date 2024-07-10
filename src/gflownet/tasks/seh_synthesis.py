import torch

from gflownet.config import Config, init_empty
from gflownet.tasks.synthesis_trainer import SynthesisTrainer
from gflownet.tasks.seh_frag import SEHTask


class SEHSynthesisTrainer(SynthesisTrainer):
    def setup_task(self):
        self.task = SEHTask(self.training_data, cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)


def main():
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.validate_every = 10
    config.num_training_steps = 1000
    config.log_dir = "./logs/debug/"
    config.env_dir = "/home/shwan/GFLOWNET_PROJECT/astb/data/envs/subsampled_10000/"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True

    config.algo.action_sampling.num_mc_sampling = 2
    config.algo.action_sampling.num_sampling_add_first_reactant = 5000
    config.algo.action_sampling.ratio_sampling_reactbi = 0.5
    config.algo.action_sampling.max_sampling_reactbi = 5000

    trial = SEHSynthesisTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
