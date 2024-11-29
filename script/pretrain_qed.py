from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(
        "RxnFlow", description="Pretrain QED Objective for QED-UniDock multi-objective optimization"
    )
    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory")
    run_cfg.add_argument(
        "-n",
        "--num_oracles",
        type=int,
        default=50000,
        help="Number of Oracles (64 molecules per oracle; default: 1000)",
    )
    run_cfg.add_argument("--env_dir", type=str, default="./data/envs/real", help="Environment Directory Path")
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
    from rxnflow.tasks.unidock_moo import UniDockMOO_Pretrainer
    from rxnflow.config import Config, init_empty

    config = init_empty(Config())
    config.env_dir = args.env_dir
    config.num_training_steps = args.num_oracles
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio
    config.checkpoint_every = 1000
    config.store_all_checkpoints = True
    config.log_dir = args.out_dir
    config.print_every = 10
    config.num_workers_retrosynthesis = 4
    if args.debug:
        config.overwrite_existing_exp = True

    trainer = UniDockMOO_Pretrainer(config)

    if args.wandb is not None:
        import wandb
        from omegaconf import OmegaConf

        wandb.init(project="rxnflow_pretrain", name=args.wandb)
        wandb.config.update({"config": OmegaConf.to_container(trainer.cfg)})
        trainer.run()
        wandb.finish()
    else:
        trainer.run()


if __name__ == "__main__":
    args = parse_args()
    run(args)
