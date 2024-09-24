import os
import sys

import wandb
from omegaconf import OmegaConf
from _exp2_constant import POCKET_DB_PATH
from gflownet.tasks.sbdd_synthesis import default_config, SBDDTrainer


def set_config(env_dir, proxy):
    proxy_model, proxy_docking, proxy_dataset = proxy
    config = default_config(env_dir, POCKET_DB_PATH, proxy_model, proxy_docking, proxy_dataset)
    config.algo.action_sampling.num_mc_sampling = 1
    config.algo.action_sampling.sampling_ratio_reactbi = 0.01
    config.algo.action_sampling.num_sampling_add_first_reactant = 12_000
    config.algo.action_sampling.max_sampling_reactbi = 12_000
    return config


def main():
    prefix = sys.argv[1]
    storage = sys.argv[2]
    env_dir = sys.argv[3]

    wandb.init(group=prefix)
    proxy = wandb.config["proxy"]
    config = set_config(env_dir, proxy)
    config.log_dir = os.path.join(storage, prefix, "-".join(proxy))

    # NOTE: Run
    trainer = SBDDTrainer(config)
    wandb.config.update({"prefix": prefix, "config": OmegaConf.to_container(trainer.cfg)})
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
