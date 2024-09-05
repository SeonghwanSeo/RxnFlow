import functools
from pathlib import Path
import argparse
import os
from typing import List

import numpy as np
from tqdm import tqdm
import multiprocessing

from rdkit import Chem
from rdkit.Chem import BondType
from gflownet.envs.synthesis.reaction import Reaction
from gflownet.envs.synthesis.building_block import get_block_features

ATOMS: list[str] = ["B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BONDS = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


def run(args, reactions: List[Reaction]):
    smiles, id = args
    mol = Chem.MolFromSmiles(smiles, replacements={"[2H]": "[H]"})

    # NOTE: Filtering Molecules with its structure
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    smi = Chem.MolToSmiles(mol)
    if smi is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    fail = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ATOMS:
            fail = True
            break
    if fail:
        return None

    for bond in mol.GetBonds():
        if bond.GetBondType() not in BONDS:
            fail = True
            break
    if fail:
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
    return smi, id, fp, desc, mask


def main(block_path: str, template_path: str, save_directory_path: str, num_cpus: int):
    save_directory = Path(save_directory_path)
    save_directory.mkdir(parents=True)
    save_template_path = save_directory / "template.txt"
    save_block_path = save_directory / "building_block.smi"
    save_mask_path = save_directory / "bb_mask.npy"
    save_desc_path = save_directory / "bb_desc.npy"
    save_fp_path = save_directory / "bb_fp_2_1024.npy"

    block_file = Path(block_path)
    assert block_file.suffix == ".sdf"

    print("Read SDF Files")
    with block_file.open() as f:
        lines = f.readlines()
    smiles_list = [lines[i].strip() for i in tqdm(range(1, len(lines))) if lines[i - 1].startswith(">  <smiles>")]
    ids = [lines[i].strip() for i in tqdm(range(1, len(lines))) if lines[i - 1].startswith(">  <id>")]

    assert len(smiles_list) == len(ids), "sdf file error, number of <smiles> and <id> should be matched"
    print("Including Mols:", len(smiles_list))
    with open(template_path, "r") as file:
        reaction_templates = file.readlines()
    reactions = [Reaction(template=t.strip()) for t in reaction_templates]  # Reaction objects
    func = functools.partial(run, reactions=reactions)

    os.system(f"cp {template_path} {save_template_path}")

    print("Run Building Blocks...")
    mask_list = []
    desc_list = []
    fp_list = []
    with open(save_block_path, "w") as w:
        for idx in tqdm(range(0, len(smiles_list), 10000)):
            chunk = list(zip(smiles_list[idx : idx + 10000], ids[idx : idx + 10000]))
            with multiprocessing.Pool(num_cpus) as pool:
                results = pool.map(func, chunk)
            for res in results:
                if res is None:
                    continue
                smiles, id, fp, desc, mask = res
                w.write(f"{smiles}\t{id}\n")
                fp_list.append(fp)
                desc_list.append(desc)
                mask_list.append(mask)

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
        help="Path to input enamine building block file (.sdf)",
    )
    parser.add_argument(
        "-t",
        "--template_path",
        type=str,
        help="Path to reaction template file",
        default="./templates/hb_edited.txt",
    )
    parser.add_argument(
        "-d",
        "--save_directory",
        type=str,
        help="Path to environment directory",
        default="./envs/enamine_all/",
    )
    parser.add_argument("--cpu", type=int, help="Num Workers")
    args = parser.parse_args()

    main(args.building_block_path, args.template_path, args.save_directory, args.cpu)
