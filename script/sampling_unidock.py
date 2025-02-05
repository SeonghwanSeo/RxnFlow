import tempfile
import os
from argparse import ArgumentParser
from pathlib import Path
import time

from rxnflow.config import Config, init_empty
from rxnflow.tasks.unidock import UniDockSampler
from utils import get_center


def parse_args():
    parser = ArgumentParser("RxnFlow", description="Inference Sampling with RxnFlow")
    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("-m", "--model_path", type=str, required=True, help="Model Checkpoitn Path")
    run_cfg.add_argument("-n", "--num_samples", type=int, required=True, help="Number of Samples")
    run_cfg.add_argument(
        "-o",
        "--out_path",
        type=str,
        required=True,
        help="Output Path (.csv | .smi). If csv, the docking score is calculated.",
    )
    run_cfg.add_argument("--env_dir", type=str, help="Environment Directory Path")
    run_cfg.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.01,
        help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.01)",
    )
    run_cfg.add_argument("--cuda", action="store_true", help="CUDA Acceleration")

    opt_cfg = parser.add_argument_group("Protein Config (overwrite)")
    opt_cfg.add_argument("-p", "--protein", type=str, required=False, help="Protein PDB Path")
    opt_cfg.add_argument("-c", "--center", nargs="+", type=float, help="Pocket Center (--center X Y Z)")
    opt_cfg.add_argument("-l", "--ref_ligand", type=str, help="Reference Ligand Path (required if center is missing)")
    opt_cfg.add_argument(
        "-s", "--size", nargs="+", type=float, help="Search Box Size (--size X Y Z)", default=(22.5, 22.5, 22.5)
    )

    run_cfg = parser.add_argument_group("Operation Config")
    return parser.parse_args()


def run(args):
    ckpt_path = Path(args.model_path)

    config = init_empty(Config())
    config.algo.num_from_policy = 100
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio
    if args.env_dir is not None:
        config.env_dir = args.env_dir
    if args.protein is not None:
        config.task.docking.protein_path = args.protein
    if args.center is not None:
        config.task.docking.center = args.center
    if args.size is not None:
        config.task.docking.size = args.size

    device = "cuda" if args.cuda else "cpu"
    save_reward = os.path.splitext(args.out_path)[1] == ".csv"

    # NOTE: Run
    with tempfile.TemporaryDirectory() as tempdir:
        config.log_dir = tempdir
        sampler = UniDockSampler(config, ckpt_path, device)
        tick_st = time.time()
        res = sampler.sample(args.num_samples, calc_reward=save_reward)
        tick_end = time.time()
    print(f"Sampling: {tick_end - tick_st:.3f} sec")
    print(f"Generated Molecules: {len(res)}")
    if save_reward:
        with open(args.out_path, "w") as w:
            w.write(",SMILES,Vina\n")
            for idx, sample in enumerate(res):
                smiles = sample["smiles"]
                vina = sample["info"]["reward"][0] * -1
                w.write(f"sample{idx},{smiles},{vina:.3f}\n")
    else:
        with open(args.out_path, "w") as w:
            for idx, sample in enumerate(res):
                w.write(f"{sample['smiles']}\tsample{idx}\n")


if __name__ == "__main__":
    args = parse_args()
    if args.ref_ligand is not None:
        args.center = get_center(args.ref_ligand)
    elif args.center is not None:
        assert len(args.center) == 3, "--center need three values: X Y Z"
    if args.size is not None:
        assert len(args.size) in [1, 3], "--center need one or three values: X Y Z"
        if len(args.size) == 1:
            v = args.size[0]
            args.size = [v, v, v]
    run(args)
