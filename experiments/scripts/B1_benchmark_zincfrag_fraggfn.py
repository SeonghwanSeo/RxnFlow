import os
import sys

from omegaconf import OmegaConf

import wandb
from rxnflow.tasks.drug_benchmark_frag import BenchmarkTrainer, Config, create_logger, init_empty

TARGET_DIR = "./data/experiments/LIT-PCBA"


def main():
    group = sys.argv[1]

    wandb.init(group=group)
    target = wandb.config["target"]
    seed = wandb.config["seed"]

    protein_path = os.path.join(TARGET_DIR, target, "protein.pdb")
    ref_ligand_path = os.path.join(TARGET_DIR, target, "ligand.mol2")

    config = init_empty(Config())
    config.desc = "LIT-PCBA Benchmark with ZINCFrag"
    config.seed = seed

    config.task.docking.protein_path = protein_path
    config.task.docking.ref_ligand_path = ref_ligand_path
    config.log_dir = os.path.join("./logs/", "benchmark", group, target, f"seed-{seed}")

    # NOTE: Run
    trainer = BenchmarkTrainer(config)
    logger = create_logger()  # non-propagate version
    wandb.config.update({"identify": f"{group}-{target}", "config": OmegaConf.to_container(trainer.cfg)})
    trainer.run(logger)
    wandb.finish()


if __name__ == "__main__":
    main()
