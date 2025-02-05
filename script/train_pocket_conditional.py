from argparse import ArgumentParser
import wandb
from omegaconf import OmegaConf
from rxnflow.tasks.multi_pocket import ProxyTrainer_MultiPocket
from rxnflow.config import Config, init_empty


def parse_args():
    parser = ArgumentParser("RxnFlow", description="Pocket-conditional GFlowNet objective training")
    opt_cfg = parser.add_argument_group("Pocket DB")
    opt_cfg.add_argument(
        "--db", type=str, default="./data/experiments/CrossDocked2020/train_db.pt", help="Pocket DB Path"
    )

    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory")
    run_cfg.add_argument(
        "-n",
        "--num_oracles",
        type=int,
        default=50000,
        help="Number of Oracles (64 molecules per oracle; default: 50000)",
    )
    run_cfg.add_argument("--env_dir", type=str, default="./data/envs/catalog", help="Environment Directory Path")
    run_cfg.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.005,
        help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.01)",
    )
    run_cfg.add_argument("--wandb", action="store_true", help="use wandb")
    run_cfg.add_argument("--debug", action="store_true", help="For debugging option")
    return parser.parse_args()


def run(args):
    config = init_empty(Config())
    config.env_dir = args.env_dir
    config.task.pocket_conditional.pocket_db = args.db
    config.task.pocket_conditional.proxy = ("TacoGFN_Reward", "QVina", "ZINCDock15M")
    config.num_training_steps = args.num_oracles
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio
    config.log_dir = args.out_dir
    config.print_every = 10
    config.checkpoint_every = 1_000
    config.store_all_checkpoints = True
    config.num_workers_retrosynthesis = 8

    if args.debug:
        config.overwrite_existing_exp = True
        config.print_every = 1

    trainer = ProxyTrainer_MultiPocket(config)

    if args.wandb:
        wandb.init()
        wandb.config.update({"config": OmegaConf.to_container(trainer.cfg)})
        trainer.run()
        wandb.finish()
    else:
        trainer.run()


if __name__ == "__main__":
    args = parse_args()
    run(args)
