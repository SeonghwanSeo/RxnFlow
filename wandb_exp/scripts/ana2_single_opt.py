import os
import sys

import wandb
from omegaconf import OmegaConf
from _exp1_constant import TARGET_CENTER, TARGET_DIR
from gflownet.tasks.unidock_moo_synthesis import UniDockMOOSynthesisTrainer
from gflownet.tasks.unidock_moo_synthesis import moo_config


def set_config(prefix, code, env_dir):
    protein_path = os.path.join(TARGET_DIR, f"{code}.pdb")
    protein_center = TARGET_CENTER[code]
    config = moo_config(env_dir, protein_path, protein_center)
    if "-all" in prefix:
        config.algo.action_sampling.sampling_ratio_reactbi = 1
        config.algo.action_sampling.num_sampling_add_first_reactant = 1_200_000
        config.algo.action_sampling.max_sampling_reactbi = 1_200_000
    else:
        config.algo.action_sampling.num_mc_sampling = 1
        config.algo.action_sampling.sampling_ratio_reactbi = 0.1
        config.algo.action_sampling.num_sampling_add_first_reactant = 1_000
        config.algo.action_sampling.max_sampling_reactbi = 1_000
    return config


def main():
    prefix = sys.argv[1]
    storage = sys.argv[2]
    env_dir = sys.argv[3]

    wandb.init(group=prefix)
    code = wandb.config["protein"]
    config = set_config(prefix, code, env_dir)
    config.log_dir = os.path.join(storage, prefix, code)

    # NOTE: Run
    trainer = UniDockMOOSynthesisTrainer(config)
    wandb.config.update({"config": OmegaConf.to_container(trainer.cfg)})
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
