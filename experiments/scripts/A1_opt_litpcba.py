from argparse import ArgumentParser
from pathlib import Path
import wandb
from omegaconf import OmegaConf

from rxnflow.tasks.unidock_moo import UniDockMOOTrainer
from rxnflow.config import Config, init_empty
from rxnflow.utils.misc import create_logger

TARGET_DIR = Path("./data/experiments/LIT-PCBA")
TARGET_CENTER = {
    "ADRB2": (-1.96, -12.27, -48.98),
    "ALDH1": (34.43, -16.88, 13.77),
    "ESR_ago": (-35.22, 4.64, 20.78),
    "ESR_antago": (17.85, 35.51, 52.49),
    "FEN1": (-16.81, -4.80, 0.62),
    "GBA": (32.44, 33.88, -19.56),
    "IDH1": (12.11, 28.09, 80.47),
    "KAT2A": (-0.11, 5.73, 10.14),
    "MAPK1": (-15.69, 14.49, 42.72),
    "MTORC1": (35.38, 49.65, 36.21),
    "OPRK1": (58.61, -24.16, -4.32),
    "PKM2": (8.64, 2.94, 10.76),
    "PPARG": (8.30, -1.02, 46.32),
    "TP53": (89.32, 91.82, -44.87),
    "VDR": (11.38, -3.12, -31.57),
}
TARGETS = list(TARGET_CENTER.keys())


if __name__ == "__main__":
    parser = ArgumentParser("RxnFlow", description="Vina-QED Optimization with GPU-accelerated UniDock")
    parser.add_argument("target", type=str, choices=TARGETS, help="LIT-PCBA target name")
    parser.add_argument("--seed", type=int, required=True, help="LIT PCBA seed")
    parser.add_argument("--wandb", type=str, help="wandb job name")
    args = parser.parse_args()

    config = init_empty(Config())
    config.env_dir = "./data/envs/catalog/"
    config.log_dir = f"./logs/litpcba/{args.target}/seed-{args.seed}"
    config.seed = args.seed
    config.print_every = 1

    config.task.docking.protein_path = TARGET_DIR / args.target / "protein.pdb"
    config.task.docking.center = TARGET_CENTER[args.target]
    config.task.docking.size = (22.5, 22.5, 22.5)

    # experiment setting
    config.num_training_steps = 1000
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0, 64]
    config.algo.action_subsampling.sampling_ratio = 0.01
    config.replay.use = False

    trainer = UniDockMOOTrainer(config)
    logger = create_logger()  # non-propagate version

    if args.wandb is not None:
        wandb.init(project="rxnflow_litpcba", name=args.wandb)
        wandb.config.update({"config": OmegaConf.to_container(trainer.cfg)})
        trainer.run(logger)
        wandb.finish()
    else:
        trainer.run()
