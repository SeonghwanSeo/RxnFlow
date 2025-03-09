from argparse import ArgumentParser

import wandb
from rxnflow.config import Config, init_empty
from rxnflow.tasks.unidock_vina import VinaTrainer


def parse_args():
    parser = ArgumentParser("RxnFlow", description="Vina optimization with GPU-accelerated UniDock")
    opt_cfg = parser.add_argument_group("Docking Config")
    opt_cfg.add_argument("-p", "--protein", type=str, required=True, help="Protein PDB Path")
    opt_cfg.add_argument("-l", "--ref_ligand", type=str, help="Reference Ligand Path (required if center is missing)")
    opt_cfg.add_argument("-c", "--center", nargs="+", type=float, help="Pocket Center (--center X Y Z)")
    opt_cfg.add_argument(
        "-s", "--size", nargs="+", type=float, help="Search Box Size (--size X Y Z)", default=(22.5, 22.5, 22.5)
    )
    opt_cfg.add_argument(
        "--search_mode",
        type=str,
        choices=["fast", "balance", "detail"],
        default="fast",
        help="UniDock Search Mode",
    )
    opt_cfg.add_argument(
        "--filter", type=str, default="lipinski", help="Drug Filter", choices=["null", "lipinski", "veber"]
    )

    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("--env_dir", type=str, default="./data/envs/catalog", help="Environment Directory Path")
    run_cfg.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory")
    run_cfg.add_argument(
        "-n",
        "--num_iterations",
        type=int,
        default=1000,
        help="Number of training iterations (64 molecules for each iterations; default: 1000)",
    )
    run_cfg.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.02,
        help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.02)",
    )
    run_cfg.add_argument("--pretrained_model_path", type=str, help="Pretrained model path")
    run_cfg.add_argument("--wandb", type=str, help="wandb job name")
    run_cfg.add_argument("--debug", action="store_true", help="For debugging option")
    return parser.parse_args()


def run(args):
    config = init_empty(Config())
    config.env_dir = args.env_dir
    config.log_dir = args.out_dir
    config.pretrained_model_path = args.pretrained_model_path

    config.print_every = 1
    config.num_training_steps = args.num_iterations
    config.algo.num_from_policy = 64
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio

    # docking info
    config.task.docking.protein_path = args.protein
    config.task.docking.ref_ligand_path = args.ref_ligand
    config.task.docking.center = args.center
    config.task.docking.size = args.size

    # drug filter
    config.task.constraint.rule = args.filter

    # set EMA factor
    if args.pretrained_model_path is None:
        config.algo.sampling_tau = 0.9
    else:
        config.algo.sampling_tau = 0.98

    # replay buffer
    config.replay.use = True
    config.replay.capacity = 64 * 200
    config.replay.warmup = 64 * 20
    config.replay.num_from_replay = 256 - 64  # batch size = 256

    if args.debug:
        config.overwrite_existing_exp = True
    if args.wandb is not None:
        wandb.init(project="rxnflow", name=args.wandb, group="unidock")

    trainer = VinaTrainer(config)
    trainer.task.vina.search_mode = args.search_mode  # set search mode
    trainer.run()
    trainer.terminate()


if __name__ == "__main__":
    args = parse_args()
    run(args)
