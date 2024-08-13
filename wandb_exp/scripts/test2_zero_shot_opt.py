import os
import sys

import wandb
from omegaconf import OmegaConf
from _exp2_constant import POCKET_DB_PATH
from gflownet.sbdd.trainer import moo_config, SynthesisTrainer_SBDD


def set_config(proxy, prefix):
    env_dir = sys.argv[3]
    config = moo_config(env_dir, POCKET_DB_PATH, proxy)
    if "-all" in prefix:
        config.algo.action_sampling.sampling_ratio_reactbi = 1
        config.algo.action_sampling.num_sampling_add_first_reactant = 1_200_000
        config.algo.action_sampling.max_sampling_reactbi = 1_200_000
    else:
        config.algo.action_sampling.num_mc_sampling = 1
        config.algo.action_sampling.sampling_ratio_reactbi = 0.01
        config.algo.action_sampling.num_sampling_add_first_reactant = 12_000
        config.algo.action_sampling.max_sampling_reactbi = 12_000
    return config


def main():
    prefix = sys.argv[1]
    storage = sys.argv[2]

    wandb.init(group=prefix)
    proxy = wandb.config["proxy"]
    config = set_config(proxy, prefix)
    config.log_dir = os.path.join(storage, prefix, proxy)

    # NOTE: Run
    trainer = SynthesisTrainer_SBDD(config)
    wandb.config.update(
        {
            "prefix": prefix,
            "config": OmegaConf.to_container(trainer.cfg),
        }
    )
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
