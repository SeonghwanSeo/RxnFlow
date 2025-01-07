import functools
import multiprocessing
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass


NUM_CPUS = 16


@dataclass
class ConversionInfo:
    name: str
    smi_file: str
    original: str
    convert: str

    @classmethod
    def from_cfg(cls, cfg: DictConfig):
        return cls(**cfg)


class Conversion:
    def __init__(self, info: ConversionInfo):
        self.cfg = info
        self.template = info.original + ">>" + info.convert
        self.rxn: ChemicalReaction = ReactionFromSmarts(self.template)
        self.rxn.Initialize()

    def is_reactant(self, mol: Chem.Mol) -> bool:
        return self.rxn.IsMoleculeReactant(mol)

    def run(self, mol: Chem.Mol) -> list[Chem.Mol]:
        res = self.rxn.RunReactants((mol,), 10)
        return list([v[0] for v in res])


def run(mol: str | Chem.Mol, rxn: Conversion) -> list[str]:
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    bricks = rxn.run(mol)
    return list(set(Chem.MolToSmiles(mol) for mol in bricks))


if __name__ == "__main__":
    # TODO: add argparse
    conversions: DictConfig = OmegaConf.load("./template/synthon.yaml")
    block_keys = list(conversions.keys())
    for key in block_keys:
        assert "-" not in str(key), "block key should not include `-`"

    block_file = Path("./building_blocks/enamine_block_clean.smi")
    save_dir = Path("./envs/real/")
    block_dir = save_dir / "blocks/"
    block_dir.mkdir(parents=True, exist_ok=True)
    # os.system("cp ./template/protocol.yaml ./template/workflow.yaml ./envs/real/")

    with block_file.open() as f:
        lines = f.readlines()[1:]
    enamine_block_list: list[str] = [ln.split()[0] for ln in lines]
    enamine_id_list: list[str] = [ln.strip().split()[1] for ln in lines]

    for i in range(len(block_keys)):
        key1 = block_keys[i]
        rxn = Conversion(conversions[key1])
        print(key1)

        func = functools.partial(run, rxn=rxn)
        with multiprocessing.Pool(NUM_CPUS) as pool:
            res = pool.map(func, enamine_block_list)

        brick_to_id: dict[str, list[str]] = {}
        for id, bricks in zip(enamine_id_list, res, strict=True):
            for smi in bricks:
                brick_to_id.setdefault(smi, []).append(id)
        with open(block_dir / f"{key1}.smi", "w") as w:
            for smi, id_list in brick_to_id.items():
                id_list.sort()
                w.write(f"{smi}\t{';'.join(id_list)}\n")
        del rxn
        del func

        brick_list = list(brick_to_id.keys())
        for j in range(i + 1, len(block_keys)):
            key2 = block_keys[j]
            print(key1, key2)
            rxn = Conversion(conversions[key2])

            func = functools.partial(run, rxn=rxn)
            with multiprocessing.Pool(NUM_CPUS) as pool:
                res = pool.map(func, brick_list)

            linker_to_id: dict[str, list[str]] = {}
            for brick, linkers in zip(brick_list, res, strict=True):
                for smi in linkers:
                    linker_to_id.setdefault(smi, []).extend(brick_to_id[brick])

            with open(block_dir / f"{key1}-{key2}.smi", "w") as w:
                for smi, ids in linker_to_id.items():
                    ids.sort()
                    w.write(f"{smi}\t{';'.join(ids)}\n")
            del rxn
            del func
