import functools
from pathlib import Path
import os

import numpy as np
from tqdm import tqdm
import multiprocessing

from rdkit import Chem
from rxnflow.envs.reaction import Reaction
from rxnflow.envs.building_block import get_block_features


def run(args, reactions: list[Reaction]):
    smiles, id = args
    mol = Chem.MolFromSmiles(smiles)

    # NOTE: Filtering Molecules which could not react with any reactions.
    unirxns = [r for r in reactions if r.num_reactants == 1]
    birxns = [r for r in reactions if r.num_reactants == 2]

    mask = np.zeros((len(birxns) * 2), dtype=np.bool_)
    for rxn_i, reaction in enumerate(birxns):
        if reaction.is_reactant(mol, 0):
            mask[rxn_i * 2 + 0] = 1
        if reaction.is_reactant(mol, 1):
            mask[rxn_i * 2 + 1] = 1
    if mask.sum() == 0:
        fail = True
        for reaction in unirxns:
            if reaction.is_reactant(mol):
                fail = False
                break
        if fail:
            return None

    fp, desc = get_block_features(mol)
    return smiles, id, fp, desc, mask


def get_block_data(block_path: str, template_path: str, save_directory_path: str, num_cpus: int):
    save_directory = Path(save_directory_path)
    save_directory.mkdir(parents=True)
    save_template_path = save_directory / "template.txt"
    save_block_path = save_directory / "building_block.smi"
    save_mask_path = save_directory / "bb_mask.npy"
    save_desc_path = save_directory / "bb_desc.npy"
    save_fp_path = save_directory / "bb_fp_2_1024.npy"

    block_file = Path(block_path)
    assert block_file.suffix == ".smi"

    print("Read SMI Files")
    with block_file.open() as f:
        lines = f.readlines()
    smi_id_list = [ln.strip().split() for ln in lines]
    print("Including Mols:", len(smi_id_list))

    with open(template_path) as file:
        reaction_templates = file.readlines()
    reactions = [Reaction(template=t.strip()) for t in reaction_templates]  # Reaction objects
    func = functools.partial(run, reactions=reactions)

    os.system(f"cp {template_path} {save_template_path}")

    print("Run Building Blocks...")
    mask_list = []
    desc_list = []
    fp_list = []
    with open(save_block_path, "w") as w:
        for idx in tqdm(range(0, len(smi_id_list), 10000)):
            chunk = smi_id_list[idx : idx + 10000]
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
