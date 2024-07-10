from pathlib import Path
from rdkit import Chem
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet.config import Config, init_empty
from gflownet.tasks.synthesis_trainer import SynthesisTrainer
from gflownet.tasks.base_task import BaseTask
from gflownet.trainer import FlatRewards, RewardScalar
from gflownet.misc.unidock.utils import unidock_scores


class UniDockTask(BaseTask):
    """Sets up a task where the reward is computed using a UniDock - TacoGFN Task."""

    def __init__(self, cfg: Config, rng: np.random.Generator, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, rng, wrap_model)
        self.root_dir = Path(cfg.task.unidock.env_dir)
        self.code = cfg.task.unidock.code
        self.protein_path = self.root_dir / "protein" / f"{self.code}.pdb"

        assert self.protein_path.exists(), f"Protein Path {self.protein_path} does not exist"

        ref_ligand_path = list((self.root_dir / "ref_ligand").glob(f"{self.code}*.sdf"))[0]
        mol = next(Chem.SDMolSupplier(str(ref_ligand_path), sanitize=False))
        x, y, z = mol.GetConformer().GetPositions().mean(0).tolist()
        self.center: tuple[float, float, float] = (x, y, z)
        self.best_molecules: List[Tuple[float, str]] = []

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        vina_scores = self.compute_unidock_rewards(mols).reshape(-1, 1)
        is_valid_t = torch.ones((len(mols),), dtype=torch.bool)
        return FlatRewards(vina_scores), is_valid_t

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        """TacoGFN Reward Function: R = Prod(-Vina Score)"""
        flat_reward = FlatRewards((flat_reward / -16.0).clip(1e-4, 10.0))
        return super()._to_reward_scalar(cond_info, flat_reward)

    def compute_unidock_rewards(self, mols: List[RDMol]) -> Tensor:
        vina_score = unidock_scores(mols, self.protein_path, self.center)
        self.update_best_molecules(mols, vina_score)
        return torch.as_tensor(vina_score)

    def update_best_molecules(self, mols: List[RDMol], scores: List[float]):
        score_smiles = [(score, Chem.MolToSmiles(mol)) for score, mol in zip(scores, mols, strict=True)]
        self.best_molecules = sorted(self.best_molecules + score_smiles)[:100]


class UniDockSynthesisTrainer(SynthesisTrainer):
    task: UniDockTask

    def setup_task(self):
        self.task = UniDockTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        info["sampled_vina_top100"] = np.mean([score for score, _ in self.task.best_molecules])
        super().log(info, index, key)


def main():
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.validate_every = 0
    config.num_training_steps = 1000
    config.log_dir = "./logs/debug/"
    config.env_dir = "/home/shwan/GFLOWNET_PROJECT/astb/data/envs/subsampled_10000/"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True

    config.algo.action_sampling.num_mc_sampling = 1
    config.algo.action_sampling.num_sampling_add_first_reactant = 5000
    config.algo.action_sampling.ratio_sampling_reactbi = 0.5
    config.algo.action_sampling.max_sampling_reactbi = 5000

    config.task.unidock.code = "14gs_A"

    trial = UniDockSynthesisTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
