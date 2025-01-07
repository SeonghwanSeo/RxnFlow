import argparse
import functools
import multiprocessing
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


class Conversion:
    def __init__(self, info: DictConfig):
        self.key = info.key
        self.template = info.original + ">>" + info.convert
        self.rxn: ChemicalReaction = ReactionFromSmarts(self.template)
        self.rxn.Initialize()

    def run(self, mol: Chem.Mol) -> list[Chem.Mol]:
        res = self.rxn.RunReactants((mol,), 10)
        return list([v[0] for v in res])


def _run_reaction(smi: str, rxn: Conversion) -> list[str]:
    mol = Chem.MolFromSmiles(smi)
    prod_mols = rxn.run(mol)
    return list(set(Chem.MolToSmiles(mol) for mol in prod_mols))


def get_block(env_dir: Path, block_file: Path, protocol_dir: Path, num_cpus: int):
    # load block
    with block_file.open() as f:
        lines = f.readlines()[1:]
    enamine_block_list: list[str] = [ln.split()[0] for ln in lines]
    enamine_id_list: list[str] = [ln.strip().split()[1] for ln in lines]

    # run conversion
    block_dir = env_dir / "blocks/"
    block_dir.mkdir(parents=True, exist_ok=True)
    conversion_config = OmegaConf.load(protocol_dir / "reactant.yaml")
    for i in tqdm(range(len(conversion_config))):
        info_i = conversion_config[i]
        rxn = Conversion(info_i)
        _func = functools.partial(_run_reaction, rxn=rxn)
        with multiprocessing.Pool(num_cpus) as pool:
            res = pool.map(_func, enamine_block_list)
        del rxn, _func

        brick_to_id: dict[str, list[str]] = {}
        for id, bricks in zip(enamine_id_list, res, strict=True):
            for smi in bricks:
                brick_to_id.setdefault(smi, []).append(id)
        if len(brick_to_id) == 0:
            continue
        with open(block_dir / f"{info_i.key}.smi", "w") as w:
            for smi, id_list in brick_to_id.items():
                w.write(f"{smi}\t{';'.join(sorted(id_list))}\n")

        brick_list = list(brick_to_id.keys())
        for j in tqdm(range(i + 1, len(conversion_config)), leave=False):
            info_j = conversion_config[j]
            rxn = Conversion(info_j)
            _func = functools.partial(_run_reaction, rxn=rxn)
            with multiprocessing.Pool(num_cpus) as pool:
                res = pool.map(_func, brick_list)
            del rxn, _func

            linker_to_id: dict[str, list[str]] = {}
            for brick, linkers in zip(brick_list, res, strict=True):
                for smi in linkers:
                    linker_to_id.setdefault(smi, []).extend(brick_to_id[brick])
            if len(linker_to_id) == 0:
                continue
            with open(block_dir / f"{info_i.key}-{info_j.key}.smi", "w") as w:
                for smi, ids in linker_to_id.items():
                    w.write(f"{smi}\t{';'.join(sorted(ids))}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert building blocks to fragments")
    parser.add_argument(
        "-d",
        "--env_dir",
        type=Path,
        help="Path of environment directory",
        default="./envs/real/",
    )
    parser.add_argument(
        "-b",
        "--building_block_path",
        type=Path,
        help="Path of building block smiles file",
        default="./building_blocks/enamine_blocks.smi",
    )
    parser.add_argument(
        "-p",
        "--protocol_dir",
        type=Path,
        help="Path of synthesis protocol",
        default="./template/real/",
    )
    parser.add_argument("--cpu", type=int, help="Num Workers")
    args = parser.parse_args()

    get_block(args.env_dir, args.building_block_path, args.protocol_dir, args.cpu)
