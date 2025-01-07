from pathlib import Path
import numpy as np
from functools import cached_property
from rdkit.Chem import rdMolDescriptors, Lipinski, Crippen
from rdkit import Chem

from rxnflow.config import Config, init_empty
from rxnflow.communication.reward import RewardModule
from rxnflow.communication.trainer import MolCommunicationTrainer
from rxnflow.utils.unidock import unidock_scores


class MolObj:
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)

    @cached_property
    def mw(self) -> float:
        return rdMolDescriptors.CalcExactMolWt(self.mol)

    @cached_property
    def tpsa(self) -> float:
        return rdMolDescriptors.CalcTPSA(self.mol)

    @cached_property
    def logp(self) -> float:
        return Crippen.MolLogP(self.mol)

    @cached_property
    def n_hba(self) -> int:
        return Lipinski.NumHAcceptor(self.mol)

    @cached_property
    def n_hbd(self) -> int:
        return Lipinski.NumHDonors(self.mol)

    @cached_property
    def n_rot_bonds(self) -> float:
        return Lipinski.NumRotatableBonds(self.mol)


class UniDockReward(RewardModule):
    def __init__(
        self,
        log_dir: str | Path,
        protein_path: str,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        threshold: float = 0.0,
        filter: str | None = "lipinski",
        ff_optimization: str | None = None,
        nproc: int = 1,
        seed: int | None = None,
    ):
        super().__init__(log_dir, verbose=True)
        self.logger.propagate = False
        assert filter in ["lipinski", "veber", None]
        assert ff_optimization in ["UFF", "MMFF", None]

        self.protein_path: str = protein_path
        self.center: tuple[float, float, float] = center
        self.size: tuple[float, float, float] = size
        self.seed: int | None = seed  # etkdg/docking seed
        self.search_mode: str = "balance"  # fast, balance, detail
        self.ff_optimization: None | str = None  # None, UFF, MMFF

        self.save_dir: Path = Path(log_dir) / "docking"
        self.save_dir.mkdir()
        self.threshold: float = threshold
        self.filter: str | None = filter

        self.batch_storage: list[tuple[str, float]] = []
        self.best_storage: list[tuple[str, float]] = []

    @property
    def objectives(self) -> list[str]:
        return ["vina"]

    def str_repr_to_object(self, obj_repr: str) -> MolObj:
        return MolObj(obj_repr)

    def compute_rewards(self, objs: list[MolObj]) -> list[list[float]]:
        rdmols = [obj.mol for obj in objs]
        vina_scores = unidock_scores(
            rdmols,
            self.protein_path,
            self.save_dir / f"{self.oracle_idx}.sdf",
            self.center,
            self.size,
            seed=1,
            search_mode=self.search_mode,
            ff_optimization=self.ff_optimization,
        )
        self.update_storage(objs, vina_scores)
        rewards = [[self.convert_docking_score(v)] for v in vina_scores]
        return rewards

    def convert_docking_score(self, reward: float) -> float:
        """Reward Scaling
        R(x) =  max(threshold - score(x), 0)
        """
        return max(self.threshold - reward, 1e-5)

    def filter_object(self, obj: MolObj) -> bool:
        if self.filter == "null":
            pass
        elif self.filter in ("lipinski", "veber"):
            if obj.mw > 500:
                return False
            if obj.n_hbd > 5:
                return False
            if obj.n_hba > 10:
                return False
            if obj.logp > 5:
                return False
            if self.filter == "veber":
                if obj.tpsa > 140:
                    return False
                if obj.n_rot_bonds > 10:
                    return False
        else:
            raise ValueError(self.filter)
        return True

    def log(self, obj_reprs: list[str], rewards: list[list[float]], is_valid: list[bool]):
        msg = "Vina Scores: \n"
        msg += f"Oracle: {self.oracle_idx}\n"
        avg_score = float(np.mean([score for _, score in self.batch_storage]))
        msg += f"  - Batch  : {avg_score:.4f}\n"
        for n in [10, 50, 100, 500, 1000]:
            topn_score = float(np.mean([score for _, score in self.best_storage[:n]]))
            msg += f"  - Top{n:<4d}: {topn_score:.4f}\n"
        self.logger.info(msg.strip())

    def update_storage(self, objs: list[MolObj], scores: list[float]):
        self.batch_storage = [(obj.smiles, score) for obj, score in zip(objs, scores, strict=True)]
        best_smi = set(smi for smi, _ in self.best_storage)
        score_smiles = [(smi, score) for smi, score in self.batch_storage if smi not in best_smi]
        self.best_storage = self.best_storage + score_smiles
        self.best_storage.sort(key=lambda v: v[1])
        self.best_storage = self.best_storage[:1000]


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    import sys

    flag = sys.argv[1]
    assert flag in ["train", "reward"]

    log_dir = "./logs/debug-unidock-communicate/"

    if flag == "train":
        config = init_empty(Config())
        config.log_dir = log_dir
        config.env_dir = "./data/envs/real"
        config.overwrite_existing_exp = True
        config.print_every = 1

        config.num_training_steps = 1000
        config.cond.temperature.sample_dist = "constant"
        config.cond.temperature.dist_params = [32.0]
        config.algo.action_subsampling.sampling_ratio = 0.1
        config.replay.use = True
        config.replay.capacity = 6_400
        config.replay.warmup = 256

        trial = MolCommunicationTrainer(config)
        trial.run()
    else:
        protein_path = "./data/examples/6oim_protein.pdb"
        center = (1.872, -8.260, -1.361)
        size = (22.5, 22.5, 22.5)
        threshold = 0.0
        filter = "lipinski"

        module = UniDockReward(log_dir, protein_path, center, size, threshold, filter)
        module.run()
