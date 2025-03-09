import os
import time
from argparse import ArgumentParser
from pathlib import Path

import gdown

from rxnflow.config import Config, init_empty
from rxnflow.tasks.multi_pocket import ProxySampler

DEFAULT_CKPT_LINK = "https://drive.google.com/uc?id=1uwvFbP0l_wNzb4riJ568Zouewhmuxvur"
DEFAULT_CKPT_PATH = "./weights/rxnflow_crossdocked_qvina.pt"
DEFAULT_CKPT_PATH = "./logs/pretrain/sbdd-uniform-0-64/model_state.pt"


def parse_args():
    parser = ArgumentParser("RxnFlow", description="(QED - Docking Proxy) Zero Shot Sampling with RxnFlow")
    opt_cfg = parser.add_argument_group("Protein Config")
    opt_cfg.add_argument("-p", "--protein", type=str, required=True, help="Protein PDB Path")
    opt_cfg.add_argument("-l", "--ref_ligand", type=str, help="Reference Ligand Path (required if center is missing)")
    opt_cfg.add_argument("-c", "--center", nargs="+", type=float, help="Pocket Center (--center X Y Z)")

    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("-n", "--num_samples", type=int, default=100, help="Number of Samples (default: 100)")
    run_cfg.add_argument("-o", "--out_path", type=str, required=True, help="Output Path (.csv | .smi)")
    run_cfg.add_argument("--env_dir", type=str, default="./data/envs/catalog", help="Environment Directory Path")
    run_cfg.add_argument("--model_path", type=str, help="Checkpoint Path")
    run_cfg.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.1,
        help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.1)",
    )
    run_cfg.add_argument("--cuda", action="store_true", help="CUDA Acceleration")
    run_cfg.add_argument("--seed", type=int, help="seed", default=1)
    return parser.parse_args()


def run(args):
    # change config from training
    config = init_empty(Config())
    config.seed = args.seed
    config.env_dir = args.env_dir
    config.algo.num_from_policy = 100
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio

    # protein path
    config.task.docking.protein_path = args.protein
    config.task.docking.ref_ligand_path = args.ref_ligand
    config.task.docking.center = args.center

    device = "cuda" if args.cuda else "cpu"
    save_reward = os.path.splitext(args.out_path)[1] == ".csv"

    # NOTE: Run
    model_path = Path(args.model_path)
    sampler = ProxySampler(config, model_path, device)
    sampler.update_temperature("uniform", [16, 64])

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
    if args.model_path is None:
        args.model_path = DEFAULT_CKPT_PATH
        if not os.path.exists(args.model_path):
            os.system("mkdir -p weights/")
            gdown.download(DEFAULT_CKPT_LINK, DEFAULT_CKPT_PATH)
    run(args)
