import os
from argparse import ArgumentParser
from pathlib import Path
import time
import gdown


DEFAULT_CKPT_LINK = "https://drive.google.com/uc?id=1uwvFbP0l_wNzb4riJ568Zouewhmuxvur"
DEFAULT_CKPT_PATH = "./weights/rxnflow_crossdocked_qvina.pt"


def parse_args():
    parser = ArgumentParser("RxnFlow", description="(QED - Docking Proxy) Zero Shot Sampling with RxnFlow")
    opt_cfg = parser.add_argument_group("Protein Config")
    opt_cfg.add_argument("-p", "--protein", type=str, required=True, help="Protein PDB Path")
    opt_cfg.add_argument("-c", "--center", nargs="+", type=float, help="Pocket Center (--center X Y Z)")
    opt_cfg.add_argument("-l", "--ref_ligand", type=str, help="Reference Ligand Path (required if center is missing)")

    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("-n", "--num_samples", type=int, default=100, help="Number of Samples (default: 100)")
    run_cfg.add_argument("-o", "--out_path", type=str, required=True, help="Output Path (.csv | .smi)")
    run_cfg.add_argument("--env_dir", type=str, default="./data/envs/real", help="Environment Directory Path")
    run_cfg.add_argument("--model_path", type=str, help="Checkpoint Path")
    run_cfg.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.01,
        help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.01)",
    )
    run_cfg.add_argument("--cuda", action="store_true", help="CUDA Acceleration")
    run_cfg.add_argument("--seed", type=int, help="seed", defualt=1)
    return parser.parse_args()


def get_center(ligand_path: str) -> tuple[float, float, float]:
    from openbabel import pybel
    import numpy as np

    extension = os.path.splitext(ligand_path)[-1][1:]
    pbmol: pybel.Molecule = next(pybel.readfile(extension, ligand_path))
    x, y, z = np.mean([atom.coords for atom in pbmol.atoms], axis=0).tolist()
    return round(x, 3), round(y, 3), round(z, 3)


def run(args):
    from rxnflow.config import Config, init_empty
    from rxnflow.tasks.multi_pocket import ProxySampler

    ckpt_path = Path(args.model_path)

    config = init_empty(Config())
    config.seed = args.seed
    config.env_dir = args.env_dir
    config.algo.global_batch_size = 100
    config.algo.action_subsampling.sampling_ratio_reactbi = args.subsampling_ratio
    config.cond.temperature.dist_params = [32, 64]

    device = "cuda" if args.cuda else "cpu"
    save_reward = os.path.splitext(args.out_path)[1] == ".csv"

    # NOTE: Run
    sampler = ProxySampler(config, ckpt_path, device)
    tick = time.time()
    sampler.set_pocket(args.protein, args.center)
    print(f"Pharmacophore Modeling: {time.time() - tick:.3f} sec")

    tick = time.time()
    res = sampler.sample(args.num_samples, calc_reward=save_reward)
    print(f"Sampling: {time.time() - tick:.3f} sec")
    print(f"Generated Molecules: {len(res)}")
    if save_reward:
        with open(args.out_path, "w") as w:
            w.write(",SMILES,QED,Proxy\n")
            for idx, sample in enumerate(res):
                smiles = sample["smiles"]
                qed = sample["info"]["reward_qed"]
                proxy = sample["info"]["reward_docking"]
                w.write(f"sample{idx},{smiles},{qed:.3f},{proxy:.3f}\n")
    else:
        with open(args.out_path, "w") as w:
            for idx, sample in enumerate(res):
                w.write(f"{sample['smiles']}\tsample{idx}\n")


if __name__ == "__main__":
    args = parse_args()
    assert (args.center is not None) or (args.ref_ligand is not None), "--center or --ref_ligand is required"
    if args.center is None:
        args.center = get_center(args.ref_ligand)
    else:
        assert len(args.center) == 3, "--center need three values: X Y Z"
    if args.model_path is None:
        args.model_path = DEFAULT_CKPT_PATH
        if not os.path.exists(args.model_path):
            os.system("mkdir -p weights/")
            gdown.download(DEFAULT_CKPT_LINK, DEFAULT_CKPT_PATH)
    run(args)
