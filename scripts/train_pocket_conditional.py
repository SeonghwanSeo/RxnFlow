from argparse import ArgumentParser

import wandb
from rxnflow.config import Config, init_empty
from rxnflow.tasks.multi_pocket import ProxyTrainer_MultiPocket


def parse_args():
    parser = ArgumentParser("RxnFlow", description="Pocket-conditional GFlowNet training")
    opt_cfg = parser.add_argument_group("Pocket DB")
    opt_cfg.add_argument(
        "--db", type=str, default="./data/experiments/CrossDocked2020/train_db.pt", help="Pocket DB Path"
    )

    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("--env_dir", type=str, default="./data/envs/catalog", help="Environment Directory Path")
    run_cfg.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory")
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
        default=64,
        help="Batch Size. Memory-variance trade-off (default: 128)",
    )
    run_cfg.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.02,
        help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.02)",
    )
    run_cfg.add_argument("--wandb", type=str, help="wandb job name")
    run_cfg.add_argument("--debug", action="store_true", help="For debugging option")
    return parser.parse_args()


def run(args):
    config = init_empty(Config())
    config.env_dir = args.env_dir
    config.log_dir = args.out_dir

    config.num_training_steps = args.num_iterations
    config.print_every = 50
    config.checkpoint_every = 1000
    config.store_all_checkpoints = True
    config.num_workers_retrosynthesis = 4

    config.task.pocket_conditional.pocket_db = args.db
    config.task.pocket_conditional.proxy = ("TacoGFN_Reward", "QVina", "ZINCDock15M")

    # === GFN parameters === #
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0.0, 64.0]

    # === Training parameters === #
    config.algo.train_random_action_prob = 0.2
    config.algo.sampling_tau = 0.9

    # pretrain -> more train and better regularization with dropout
    config.model.dropout = 0.1

    # training batch size & subsampling size
    # cost-variance trade-off parameters
    config.algo.num_from_policy = args.batch_size
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio

    # training learning rate
    config.opt.learning_rate = 1e-4
    config.opt.lr_decay = 10_000
    config.algo.tb.Z_learning_rate = 1e-2
    config.algo.tb.Z_lr_decay = 20_000

    if args.debug:
        config.overwrite_existing_exp = True
        config.print_every = 1
    if args.wandb:
        wandb.init(project="rxnflow", name=args.wandb, group="pocket-conditional")

    trainer = ProxyTrainer_MultiPocket(config)
    trainer.run()
    trainer.terminate()


if __name__ == "__main__":
    args = parse_args()
    run(args)
