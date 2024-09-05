import argparse
import os
from pathlib import Path
import numpy as np
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample building blocks")
    parser.add_argument(
        "-r",
        "--root_dir",
        type=str,
        help="Path to root (entire) environment directory",
        default="./envs/enamine_all",
    )
    parser.add_argument("-d", "--save_dir", type=str, help="Path to environment directory")
    parser.add_argument("-n", "--num_samples", type=int, help="Number of building blocks to subsample")
    parser.add_argument("--seed", type=int, help="Random Seed", default=1)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    template_path = root_dir / "template.txt"
    block_path = root_dir / "building_block.smi"
    mask_path = root_dir / "bb_mask.npy"
    fp_path = root_dir / "bb_fp_2_1024.npy"
    desc_path = root_dir / "bb_desc.npy"

    with open(block_path) as f:
        lines = f.readlines()

    print(f"get subset with randomly selected {args.num_samples} blocks with seed {args.seed}")
    random.seed(args.seed)
    indices = list(range(len(lines)))
    random.shuffle(indices)
    indices = indices[: args.num_samples]
    indices.sort()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=False)
    save_template_path = save_dir / "template.txt"
    save_block_path = save_dir / "building_block.smi"
    save_mask_path = save_dir / "bb_mask.npy"
    save_fp_path = save_dir / "bb_fp_2_1024.npy"
    save_desc_path = save_dir / "bb_desc.npy"
    os.system(f"cp {template_path} {save_template_path}")

    mask = np.load(mask_path)
    fp = np.load(fp_path)
    desc = np.load(desc_path)

    with save_block_path.open("w") as w:
        for idx in indices:
            w.write(lines[idx])
    np.save(save_mask_path, mask[:, :, indices])
    np.save(save_fp_path, fp[indices])
    np.save(save_desc_path, desc[indices])
