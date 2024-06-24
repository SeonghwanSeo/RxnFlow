from pathlib import Path
import argparse
import os

import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import BondType
from gflownet.envs.synthesis.utils import Reaction

ATOMS: list[str] = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B"]
BONDS = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


def main(
    block_path: str,
    template_path: str,
    save_directory_path: str,
):
    save_directory = Path(save_directory_path)
    save_directory.mkdir(parents=True)
    save_template_path = save_directory / "template.txt"
    save_block_path = save_directory / "building_block.smi"
    save_mask_path = save_directory / "precompute_bb_mask.npy"

    block_file = Path(block_path)
    assert block_file.suffix == ".sdf"

    print("Read SDF Files")
    with block_file.open() as f:
        lines = f.readlines()
    smiles = [lines[i].strip() for i in tqdm(range(1, len(lines))) if lines[i - 1].startswith(">  <smiles>")]
    ids = [lines[i].strip() for i in tqdm(range(1, len(lines))) if lines[i - 1].startswith(">  <id>")]

    assert len(smiles) == len(ids), "sdf file error, number of <smiles> and <id> should be matched"
    print("Including Mols:", len(smiles))
    with open(template_path, "r") as file:
        reaction_templates = file.readlines()
    reactions = [Reaction(template=t.strip()) for t in reaction_templates]  # Reaction objects
    bimolecular_reactions = [r for r in reactions if r.num_reactants == 2]

    os.system(f"cp {template_path} {save_template_path}")

    print("Run Building Blocks...")
    t = 0
    all_mask = np.zeros((len(smiles), len(bimolecular_reactions), 2), dtype=np.bool_)
    with open(save_block_path, "w") as w:
        for _smi, id in zip(tqdm(smiles), ids, strict=True):
            mol = Chem.MolFromSmiles(_smi, replacements={"[2H]": "[H]"})

            # NOTE: Filtering Molecules with its structure
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except:
                continue

            smi = Chem.MolToSmiles(mol)
            if smi is None:
                continue

            fail = False
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ATOMS:
                    fail = True
                    break
            if fail:
                continue

            for bond in mol.GetBonds():
                if bond.GetBondType() not in BONDS:
                    fail = True
                    break
            if fail:
                continue

            # NOTE: Filtering Molecules which could not react with any reactions.
            mask = np.zeros((len(bimolecular_reactions), 2), dtype=np.bool_)
            for rxn_i, reaction in enumerate(bimolecular_reactions):
                reactants = reaction.rxn.GetReactants()
                if mol.HasSubstructMatch(reactants[0]):
                    mask[rxn_i, 0] = 1
                if mol.HasSubstructMatch(reactants[1]):
                    mask[rxn_i, 1] = 1
            if mask.sum() == 0:
                continue
            all_mask[t] = mask

            w.write(f"{smi}\t{id}\n")
            t += 1

    all_mask = all_mask[:t].transpose((1, 0, 2))
    print(f"Saving precomputed masks to of shape={all_mask.shape} to {save_mask_path}")
    np.save(save_mask_path, all_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample building blocks")
    parser.add_argument(
        "-b", "--building_block_path", type=str, help="Path to input enamine building block file (.sdf)"
    )
    parser.add_argument("-t", "--template_path", type=str, help="Path to reaction template file")
    parser.add_argument("-d", "--save_directory", type=str, help="Path to environment directory")
    args = parser.parse_args()

    main(args.building_block_path, args.template_path, args.save_directory)
