import functools
from pathlib import Path
import argparse

import numpy as np
from tqdm import tqdm

from rdkit import Chem
from gflownet.envs.synthesis.reaction import Reaction
from gflownet.envs.synthesis.building_block import get_block_features

REACTION_TEMPLATES = [
    "[Cl,OH,O-:3][C$(C(=O)([CX4,c])),C$([CH](=O)):2]=[O:4].[N$([NH2,NH3+1]([CX4,c])),N$([NH]([CX4,c])([CX4,c])):6]>>[N+0:6]-[C:2]=[O:4]",
    "[OH+0,O-:5]-[C:3](=[O:4])-[C$([CH]([CX4])),C$([CH2]):2]>>[OH+0,O-:5]-[C:3](=[O:4])-[C:2]([Cl:6])",
]
NUM_BLOCKS = 10000


def run(args, reactions: list[Reaction]):
    smiles, id = args
    mol = Chem.MolFromSmiles(smiles, replacements={"[2H]": "[H]"})
    if mol is None:
        return None

    # NOTE: Filtering Molecules which could not react with any reactions.
    unimolecular_reactions = [r for r in reactions if r.num_reactants == 1]
    bimolecular_reactions = [r for r in reactions if r.num_reactants == 2]

    mask = np.zeros((len(bimolecular_reactions), 2), dtype=np.bool_)
    for rxn_i, reaction in enumerate(bimolecular_reactions):
        if reaction.is_reactant_first(mol):
            mask[rxn_i, 0] = 1
        if reaction.is_reactant_second(mol):
            mask[rxn_i, 1] = 1
    if mask.sum() == 0:
        fail = True
        for reaction in unimolecular_reactions:
            if reaction.is_reactant(mol):
                fail = False
                break
        if fail:
            return None

    fp, desc = get_block_features(mol, 2, 1024)
    return smiles, id, fp, desc, mask


def main(block_path: str):
    save_directory = Path("./envs/toy/")
    save_directory.mkdir(parents=True)
    save_template_path = save_directory / "template.txt"
    save_block_path = save_directory / "building_block.smi"
    save_mask_path = save_directory / "bb_mask.npy"
    save_fp_path = save_directory / "bb_fp_2_1024.npy"
    save_desc_path = save_directory / "bb_desc.npy"

    block_file = Path(block_path)
    assert block_file.suffix == ".smi"

    print("Read SMI Files")
    with block_file.open() as f:
        lines = f.readlines()
    smi_id_list = [ln.strip().split() for ln in lines]
    print("Including Mols:", len(smi_id_list))

    reactions = [Reaction(template=t.strip()) for t in REACTION_TEMPLATES]  # Reaction objects
    func = functools.partial(run, reactions=reactions)

    with open(save_template_path, "w") as w:
        for template in REACTION_TEMPLATES:
            w.write(template + "\n")

    print("Run Building Blocks...")
    mask_list = []
    desc_list = []
    fp_list = []
    with open(save_block_path, "w") as w:
        for idx in tqdm(range(0, len(smi_id_list), 10000)):
            chunk = smi_id_list[idx : idx + 10000]
            results = map(func, chunk)
            for res in results:
                if res is None:
                    continue
                smiles, id, fp, desc, mask = res
                w.write(f"{smiles}\t{id}\n")
                fp_list.append(fp)
                desc_list.append(desc)
                mask_list.append(mask)
                if len(fp_list) >= NUM_BLOCKS:
                    break
            if len(fp_list) >= NUM_BLOCKS:
                break

    all_mask = np.stack(mask_list, -1)
    building_block_descs = np.stack(desc_list, 0)
    building_block_fps = np.stack(fp_list, 0)
    print(f"Saving precomputed masks to of shape={all_mask.shape} to {save_mask_path}")
    print(f"Saving precomputed RDKit Descriptors to of shape={building_block_descs.shape} to {save_desc_path}")
    print(f"Saving precomputed Morgan/MACCS Fingerprints to of shape={building_block_fps.shape} to {save_fp_path}")
    np.save(save_mask_path, all_mask)
    np.save(save_desc_path, building_block_descs)
    np.save(save_fp_path, building_block_fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample building blocks")
    parser.add_argument(
        "-b",
        "--building_block_path",
        type=str,
        help="Path to input enamine building block file (.smi)",
        default="./building_blocks/Enamine_Building_Blocks_1309385cmpd_20240610.smi",
    )
    args = parser.parse_args()

    main(args.building_block_path)
