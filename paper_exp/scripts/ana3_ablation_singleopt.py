import os
import sys

import wandb
from omegaconf import OmegaConf
from _exp1_constant import TARGET_CENTER, TARGET_DIR
from gflownet.tasks.unidock_moo_synthesis import UniDockMOOSynthesisTrainer
from gflownet.tasks.unidock_moo_synthesis import moo_config


ENV_DIR = {
    100: "./data/envs/ablation/subsampled_100/",
    1000: "./data/envs/ablation/subsampled_1k/",
    10000: "./data/envs/ablation/subsampled_10k/",
    100000: "./data/envs/ablation/subsampled_100k/",
    1000000: "./data/envs/ablation/subsampled_1M/",
}
SAMPLING_RATIO = {
    100: 1.0,
    1000: 1.0,
    10000: 1.0,
    100000: 0.1,
    1000000: 0.01,
}


def set_config(code, num_blocks):
    protein_path = os.path.join(TARGET_DIR, f"{code}.pdb")
    protein_center = TARGET_CENTER[code]

    env_dir = ENV_DIR[num_blocks]
    sampling_ratio = SAMPLING_RATIO[num_blocks]
    config = moo_config(env_dir, protein_path, protein_center)

    config.algo.action_sampling.num_mc_sampling = 1
    config.algo.action_sampling.sampling_ratio_reactbi = sampling_ratio
    config.algo.action_sampling.num_sampling_add_first_reactant = int(num_blocks * sampling_ratio)
    config.algo.action_sampling.max_sampling_reactbi = int(num_blocks * sampling_ratio)
    return config


def main():
    code = sys.argv[1]
    storage = sys.argv[2]

    wandb.init(group=code)
    num_blocks = wandb.config["num_blocks"]
    trial = wandb.config["trial"]
    config = set_config(code, num_blocks)
    config.log_dir = os.path.join(storage, code, f"trial-{trial}", str(num_blocks))

    # NOTE: Run
    trainer = UniDockMOOSynthesisTrainer(config)
    wandb.config.update({"config": OmegaConf.to_container(trainer.cfg)})
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
