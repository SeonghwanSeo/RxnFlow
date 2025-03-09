from argparse import ArgumentParser

import wandb
from rxnflow.config import Config, init_empty
from rxnflow.tasks.seh import SEHTrainer


def parse_args():
    parser = ArgumentParser("RxnFlow", description="SEH Proxy optimization")
    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory")
    run_cfg.add_argument(
        "-n",
        "--num_iterations",
        type=int,
        default=10_000,
        help="Number of training iterations (default: 10,000)",
    )
    run_cfg.add_argument("--env_dir", type=str, default="./data/envs/catalog", help="Environment Directory Path")
    run_cfg.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.01,
        help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.01)",
    )
    run_cfg.add_argument("--wandb", type=str, help="wandb job name")
    run_cfg.add_argument("--debug", action="store_true", help="For debugging option")
    return parser.parse_args()


def run(args):
    config = init_empty(Config())
    config.env_dir = args.env_dir
    config.log_dir = args.out_dir
    config.print_every = 10
    config.num_training_steps = args.num_iterations
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio

    config.opt.learning_rate = 1e-4
    config.opt.lr_decay = 2000
    config.algo.tb.Z_learning_rate = 1e-2
    config.algo.tb.Z_lr_decay = 5000

    if args.debug:
        config.overwrite_existing_exp = True
    if args.wandb is not None:
        wandb.init(project="rxnflow", name=args.wandb, group="seh")

    trainer = SEHTrainer(config)
    trainer.run()
    trainer.terminate()


if __name__ == "__main__":
    args = parse_args()
    run(args)
