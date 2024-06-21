from pathlib import Path
import argparse
import os

import numpy as np
from tqdm import tqdm

from rdkit import Chem
from gflownet.envs.synthesis.utils import Reaction


def precompute_bb_masks(building_block_path, template_path, save_directory):
    print("Precomputing building blocks masks for each reaction and reactant position...")
    with open(template_path, "r") as file:
        REACTION_TEMPLATES = file.readlines()
    with open(building_block_path, "r") as file:
        lines = file.readlines()
        BUILDING_BLOCKS = [ln.split()[0] for ln in lines]

    reactions = [Reaction(template=t.strip()) for t in REACTION_TEMPLATES]  # Reaction objects
    bimolecular_reactions = [r for r in reactions if r.num_reactants == 2]
    building_block_mols = [Chem.MolFromSmiles(bb) for bb in BUILDING_BLOCKS]

    masks = np.zeros((len(bimolecular_reactions), len(building_block_mols), 2), dtype=np.bool_)
    for rxn_i, reaction in enumerate(tqdm(bimolecular_reactions)):
        reactants = reaction.rxn.GetReactants()
        for bb_j, bb in enumerate(building_block_mols):
            if bb.HasSubstructMatch(reactants[0]):
                masks[rxn_i, bb_j, 0] = 1
            if bb.HasSubstructMatch(reactants[1]):
                masks[rxn_i, bb_j, 1] = 1

    save_directory = Path(save_directory)
    save_template_path = save_directory / "template.txt"
    save_block_path = save_directory / "building_block.smi"
    save_mask_path = save_directory / "precompute_bb_mask.npy"

    save_directory.mkdir(parents=True)
    os.system(f"cp {building_block_path} {save_block_path}")
    os.system(f"cp {template_path} {save_template_path}")

    print(f"Saving precomputed masks to of shape={masks.shape} to {save_mask_path}")
    np.save(save_mask_path, masks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample building blocks")
    parser.add_argument("-b", "--block_path", type=str, help="Path to input building block smi file")
    parser.add_argument("-t", "--template_path", type=str, help="Path to reaction template file")
    parser.add_argument("-d", "--save_directory", type=str, help="Path to environment directory")
    args = parser.parse_args()

    precompute_bb_masks(args.block_path, args.template_path, args.save_directory)
