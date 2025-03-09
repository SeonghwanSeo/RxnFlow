from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf

import wandb
from rxnflow.config import Config, init_empty
from rxnflow.tasks.drug_benchmark_moo import BenchmarkTrainer
from rxnflow.utils.misc import create_logger

TARGET_DIR = Path("./data/experiments/LIT-PCBA")
TARGETS = [
    "ADRB2",
    "ALDH1",
    "ESR_ago",
    "ESR_antago",
    "FEN1",
    "GBA",
    "IDH1",
    "KAT2A",
    "MAPK1",
    "MTORC1",
    "OPRK1",
    "PKM2",
    "PPARG",
    "TP53",
    "VDR",
]


if __name__ == "__main__":
    parser = ArgumentParser("RxnFlow", description="Vina-QED Optimization with GPU-accelerated UniDock")
    parser.add_argument("target", type=str, choices=TARGETS, help="LIT-PCBA target name")
    parser.add_argument("--seed", type=int, required=True, help="LIT PCBA seed")
    parser.add_argument("--wandb", type=str, help="wandb job name")
    args = parser.parse_args()

    config = init_empty(Config())
    config.env_dir = "./data/envs/enamine_catalog/"
    config.log_dir = f"./logs/benchmark_moo/{args.target}/seed-{args.seed}"
    config.seed = args.seed
    config.print_every = 1

    config.task.docking.protein_path = TARGET_DIR / args.target / "protein.pdb"
    config.task.docking.ref_ligand_path = TARGET_DIR / args.target / "ref_ligand.mol2"
    config.task.docking.size = (22.5, 22.5, 22.5)

    # experiment setting
    config.num_training_steps = 1000
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0, 64]
    config.algo.action_subsampling.sampling_ratio = 0.02
    config.replay.use = False

    trainer = BenchmarkTrainer(config)
    logger = create_logger()  # non-propagate version

    if args.wandb is not None:
        wandb.init(project="rxnflow_benchmark", name=args.wandb, group="moo")
        wandb.config.update({"config": OmegaConf.to_container(trainer.cfg)})
        trainer.run(logger)
        wandb.finish()
    else:
        trainer.run()
