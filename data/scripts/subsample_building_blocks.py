import argparse
import random
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm

ATOMS: list[str] = [
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "B",
    "Sn",
    "Ca",
    "Na",
    "Ba",
    "Zn",
    "Rh",
    "Ag",
    "Li",
    "Yb",
    "K",
    "Fe",
    "Cs",
    "Bi",
    "Pd",
    "Cu",
    "Si",
]


def parse_bool(b):
    if b.lower() in ["true", "t", "1"]:
        return True
    elif b.lower() in ["false", "f", "0"]:
        return False
    else:
        raise ValueError("Invalid boolean value")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample building blocks")
    parser.add_argument("-f", "--filepath", type=str, help="Path to input building blocks")
    parser.add_argument("-n", "--num_samples", type=int, help="Number of building blocks to subsample", default=100_000)
    parser.add_argument(
        "--random",
        type=parse_bool,
        help="If true sample building blocks uniformly at random, otherwise take the first n.",
        default=True,
    )
    args = parser.parse_args()

    filepath = Path(args.filepath)
    root_dir = filepath.parent
    assert filepath.suffix in (".smi", ".sdf")

    if filepath.suffix == ".smi":
        with filepath.open() as f:
            bb_list = [ln.strip().split() for ln in f.readlines()]
    else:
        assert filepath.suffix == ".sdf"
        with filepath.open() as f:
            lines = f.readlines()
        smiles = [lines[i].strip() for i in tqdm(range(1, len(lines))) if lines[i - 1].startswith(">  <smiles>")]
        ids = [lines[i].strip() for i in tqdm(range(1, len(lines))) if lines[i - 1].startswith(">  <id>")]

        print("num smiles:", len(smiles), "num_ids:", len(ids))

        smiles_filepath = root_dir / f"{filepath.stem}.smi"
        with smiles_filepath.open("w") as w:
            for _smi, id in tqdm(zip(smiles, ids, strict=True)):
                mol = Chem.MolFromSmiles(_smi, replacements={"[2H]": "[H]"})
                if mol is None:
                    continue
                fail = False
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() not in ATOMS:
                        fail = True
                        break
                if fail:
                    continue
                smi = Chem.MolToSmiles(mol)
                if smi is None:
                    continue
                w.write(f"{smi}\t{id}\n")
        with (root_dir / f"{filepath.stem}.smi").open() as f:
            bb_list = [ln.strip().split() for ln in f.readlines()]
        print("num valid blocks:", len(bb_list))
        print(f"Save to {smiles_filepath}")

    if args.random:
        random.seed(0)
        bb_list = random.sample(bb_list, args.num_samples)
        suffix = "subsampled"
    else:
        bb_list = bb_list[: args.num_samples]
        suffix = "first"

    sampled_filename = filepath.stem + f"_{suffix}_{args.num_samples}.smi"
    with (root_dir / sampled_filename).open("w") as w:
        for smi, id in bb_list:
            w.write(f"{smi}\t{id}\n")
