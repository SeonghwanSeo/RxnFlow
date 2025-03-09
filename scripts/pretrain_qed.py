from argparse import ArgumentParser

import wandb
from rxnflow.config import Config, init_empty
from rxnflow.tasks.qed import QEDTrainer


def parse_args():
    parser = ArgumentParser("RxnFlow", description="QED Pretraining")
    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("--env_dir", type=str, default="./data/envs/catalog", help="Environment Directory Path")
    run_cfg.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory")
    run_cfg.add_argument(
        "--temperature",
        type=str,
        default="uniform-0-64",
        help="temperature setting (e.g., constant-32 ; uniform-0-64(default))",
    )
    run_cfg.add_argument(
        "-n",
        "--num_iterations",
        type=int,
        default=50_000,
        help="Number of training iterations (default: 50,000)",
    )
    run_cfg.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch Size. Memory-variance trade-off (default: 128)",
    )
    run_cfg.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.05,
        help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.05)",
    )
    run_cfg.add_argument("--wandb", type=str, help="wandb job name")
    run_cfg.add_argument("--debug", action="store_true", help="For debugging option")
    return parser.parse_args()


def run(args):
    from _utils import parse_temperature

    config = init_empty(Config())
    config.env_dir = args.env_dir
    config.log_dir = args.out_dir

    config.num_training_steps = args.num_iterations
    config.print_every = 50
    config.checkpoint_every = 500
    config.store_all_checkpoints = True
    config.num_workers_retrosynthesis = 4

    # === GFN parameters === #
    sample_dist, dist_params = parse_temperature(args.temperature)
    config.cond.temperature.sample_dist = sample_dist
    config.cond.temperature.dist_params = dist_params

    # === Training parameters === #
    # we set high random action prob
    # so, we do not use Double-GFN
    config.algo.train_random_action_prob = 0.5
    config.algo.sampling_tau = 0.0

    # pretrain -> more train and better regularization with dropout
    config.model.dropout = 0.1

    # training batch size & subsampling size
    # cost-variance trade-off parameters
    config.algo.num_from_policy = args.batch_size
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio

    # replay buffer
    # each training batch: 128 mols from policy and 128 mols from buffer
    config.replay.use = True
    config.replay.warmup = args.batch_size * 10
    config.replay.capacity = args.batch_size * 200

    # training learning rate
    config.opt.learning_rate = 1e-4
    config.opt.lr_decay = 10_000
    config.algo.tb.Z_learning_rate = 1e-2
    config.algo.tb.Z_lr_decay = 20_000

    if args.debug:
        config.overwrite_existing_exp = True
        config.print_every = 1
    if args.wandb is not None:
        wandb.init(project="rxnflow", name=args.wandb, group="qed-pretrain")

    trainer = QEDTrainer(config)
    trainer.run()
    trainer.terminate()


if __name__ == "__main__":
    args = parse_args()
    run(args)
