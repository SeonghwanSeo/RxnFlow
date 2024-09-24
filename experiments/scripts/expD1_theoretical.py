import os
import sys

import wandb
from omegaconf import OmegaConf
from gflownet.config import Config, init_empty
from gflownet.tasks.analysis_toy import ToyQEDTrainer


def set_config(env_dir, sampling_ratio, num_mc_samples):
    config = init_empty(Config())
    config.env_dir = env_dir

    config.algo.action_sampling.num_mc_sampling = num_mc_samples
    config.algo.action_sampling.sampling_ratio_reactbi = sampling_ratio
    config.algo.action_sampling.max_sampling_reactbi = 10_000
    config.algo.action_sampling.num_sampling_add_first_reactant = int(10_000 * sampling_ratio)
    return config


def main():
    prefix = sys.argv[1]
    storage = sys.argv[2]
    env_dir = sys.argv[3]

    wandb.init(group=prefix)
    rev_sampling_ratio, num_mc_samples = wandb.config["subsampling_params"]
    sampling_ratio = 1 / rev_sampling_ratio

    wandb.config["sampling_ratio"] = sampling_ratio
    wandb.config["num_mc_samples"] = num_mc_samples

    config = set_config(env_dir, sampling_ratio, num_mc_samples)
    config.log_dir = os.path.join(storage, prefix, f"mc-{num_mc_samples}-sr-{rev_sampling_ratio}")

    # NOTE: Run
    trainer = ToyQEDTrainer(config)
    wandb.config.update({"config": OmegaConf.to_container(trainer.cfg)})
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
