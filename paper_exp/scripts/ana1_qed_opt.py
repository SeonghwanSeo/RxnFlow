import os
import sys

import wandb
from omegaconf import OmegaConf
from gflownet.config import Config, init_empty
from gflownet.tasks.analysis_qed import QEDSynthesisTrainer


def set_config(prefix, env_dir, beta):
    config = init_empty(Config())
    config.env_dir = env_dir
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [beta]
    config.num_training_steps = 5000
    config.print_every = 5

    if "-all" in prefix:
        config.algo.action_sampling.sampling_ratio_reactbi = 1
        config.algo.action_sampling.num_sampling_add_first_reactant = 1_200_000
        config.algo.action_sampling.max_sampling_reactbi = 1_200_000
    else:
        config.algo.action_sampling.num_mc_sampling = 1
        config.algo.action_sampling.sampling_ratio_reactbi = 0.02
        config.algo.action_sampling.num_sampling_add_first_reactant = 10_000
        config.algo.action_sampling.max_sampling_reactbi = 10_000
    return config


def main():
    prefix = sys.argv[1]
    storage = sys.argv[2]
    env_dir = sys.argv[3]

    wandb.init(group=prefix)
    beta = wandb.config["sampling_beta"]
    config = set_config(prefix, env_dir, beta)
    config.log_dir = os.path.join(storage, prefix, f"beta-{beta}")

    # NOTE: Run
    trainer = QEDSynthesisTrainer(config)
    wandb.config.update({"config": OmegaConf.to_container(trainer.cfg)})
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
