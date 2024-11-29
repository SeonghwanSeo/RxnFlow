import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser("RxnFlow", description="QED-UniDock Optimization with RxnFlow")
    opt_cfg = parser.add_argument_group("Protein Config")
    opt_cfg.add_argument("-p", "--protein", type=str, required=True, help="Protein PDB Path")
    opt_cfg.add_argument("-c", "--center", nargs="+", type=float, help="Pocket Center (--center X Y Z)")
    opt_cfg.add_argument("-l", "--ref_ligand", type=str, help="Reference Ligand Path (required if center is missing)")
    opt_cfg.add_argument(
        "-s", "--size", nargs="+", type=float, help="Search Box Size (--size X Y Z)", default=(22.5, 22.5, 22.5)
    )

    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory")
    run_cfg.add_argument(
        "-n",
        "--num_oracles",
        type=int,
        default=1000,
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


def get_center(ligand_path: str) -> tuple[float, float, float]:
    from openbabel import pybel
    import numpy as np

    extension = os.path.splitext(ligand_path)[-1][1:]
    pbmol: pybel.Molecule = next(pybel.readfile(extension, ligand_path))
    x, y, z = np.mean([atom.coords for atom in pbmol.atoms], axis=0).tolist()
    return round(x, 3), round(y, 3), round(z, 3)


def run(args):
    from rxnflow.tasks.unidock_moo import UniDockMOOTrainer
    from rxnflow.config import Config, init_empty

    config = init_empty(Config())
    config.env_dir = args.env_dir
    config.task.docking.protein_path = args.protein
    config.task.docking.center = tuple(args.center)
    config.task.docking.size = tuple(args.size)
    config.num_training_steps = args.num_oracles
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio
    config.log_dir = args.out_dir
    config.print_every = 1
    if args.debug:
        config.overwrite_existing_exp = True

    trainer = UniDockMOOTrainer(config)

    if args.wandb is not None:
        import wandb
        from omegaconf import OmegaConf

        wandb.init(project="rxnflow_unidock", name=args.wandb)
        wandb.config.update({"config": OmegaConf.to_container(trainer.cfg)})
        trainer.run()
        wandb.finish()
    else:
        trainer.run()


if __name__ == "__main__":
    args = parse_args()
    assert (args.center is not None) or (args.ref_ligand is not None), "--center or --ref_ligand is required"
    if args.center is None:
        args.center = get_center(args.ref_ligand)
    else:
        assert len(args.center) == 3, "--center need three values: X Y Z"
    run(args)
