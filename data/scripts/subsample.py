import argparse
import os
from pathlib import Path
import numpy as np
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample building blocks")
    parser.add_argument("-r", "--root_dir", type=str, help="Path to input enamine building block file (.sdf)")
    parser.add_argument("-d", "--save_dir", type=str, help="Path to environment directory")
    parser.add_argument("-n", "--num_samples", type=int, help="Number of building blocks to subsample", default=10_000)
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sampling building blocks uniformly at random, otherwise take the first n.",
    )
    parser.add_argument("--seed", type=int, help="Random Seed", default=1)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    template_path = root_dir / "template.txt"
    block_path = root_dir / "building_block.smi"
    mask_path = root_dir / "precompute_bb_mask.npy"

    with open(block_path) as f:
        lines = f.readlines()
    mask = np.load(mask_path)
    assert mask.any(axis=(0, 2)).all()

    if args.random:
        print(f"get subset with randomly selected {args.num_samples} blocks with seed {args.seed}")
        random.seed(args.seed)
        indices = list(range(len(lines)))
        random.shuffle(indices)
        indices = indices[: args.num_samples]
        indices.sort()
    else:
        print(f"get subset with first {args.num_samples} blocks")
        indices = list(range(args.num_samples))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True)
    save_template_path = save_dir / "template.txt"
    save_block_path = save_dir / "building_block.smi"
    save_mask_path = save_dir / "precompute_bb_mask.npy"
    os.system(f"cp {template_path} {save_template_path}")

    with save_block_path.open("w") as w:
        for idx in indices:
            w.write(lines[idx])
    np.save(save_mask_path, mask[:, indices, :])
