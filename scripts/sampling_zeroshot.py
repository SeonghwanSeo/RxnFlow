import os
import time
from argparse import ArgumentParser

from rxnflow.config import Config, init_empty
from rxnflow.tasks.multi_pocket import ProxySampler
from rxnflow.utils.download import download_pretrained_weight

DEFAULT_CKPT = "qvina-unif-0-64"


def parse_args():
    parser = ArgumentParser("RxnFlow", description="(QED - Docking Proxy) Zero Shot Sampling with RxnFlow")
    opt_cfg = parser.add_argument_group("Protein Config")
    opt_cfg.add_argument("-p", "--protein", type=str, required=True, help="Protein PDB Path")
    opt_cfg.add_argument("-l", "--ref_ligand", type=str, help="Reference Ligand Path (required if center is missing)")
    opt_cfg.add_argument("-c", "--center", nargs="+", type=float, help="Pocket Center (--center X Y Z)")

    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("--model_path", type=str, help="Checkpoint Path", default="qvina-unif-0-64")
    run_cfg.add_argument("-n", "--num_samples", type=int, default=100, help="Number of Samples (default: 100)")
    run_cfg.add_argument("-o", "--out_path", type=str, required=True, help="Output Path (.csv | .smi)")
    run_cfg.add_argument("--env_dir", type=str, default="./data/envs/catalog", help="Environment Directory Path")
    run_cfg.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.1,
        help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.1)",
    )
    run_cfg.add_argument(
        "--temperature",
        type=str,
        default="uniform-16-64",
        help="temperature setting (e.g., uniform-16-64(default), uniform-32-64, ...)",
    )
    run_cfg.add_argument("--cuda", action="store_true", help="CUDA Acceleration")
    run_cfg.add_argument("--seed", type=int, help="seed", default=1)
    return parser.parse_args()


def run(args):
    from _utils import parse_temperature

    # change config from training
    config = init_empty(Config())
    config.seed = args.seed
    config.env_dir = args.env_dir
    config.algo.num_from_policy = 100
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio

    device = "cuda" if args.cuda else "cpu"
    save_reward = os.path.splitext(args.out_path)[1] == ".csv"

    # create sampler
    model_path = download_pretrained_weight(args.model_path)
    sampler = ProxySampler(config, model_path, device)
    sample_dist, dist_params = parse_temperature(args.temperature)
    sampler.update_temperature(sample_dist, dist_params)

    # set binding site
    sampler.set_pocket(args.protein, args.center, args.ref_ligand)

    # run
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
                proxy = sample["info"]["reward_vina"]
                w.write(f"sample{idx},{smiles},{qed:.3f},{proxy:.3f}\n")
    else:
        with open(args.out_path, "w") as w:
            for idx, sample in enumerate(res):
                w.write(f"{sample['smiles']}\tsample{idx}\n")


if __name__ == "__main__":
    args = parse_args()
    run(args)
