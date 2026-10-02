"""Minimal local reward injection example."""

import torch
from rdkit import Chem

from rxnflow.config import Config
from rxnflow.reward import RewardFunction
from rxnflow.trainer import RxnFlowTrainer


class HeavyAtomReward(RewardFunction):
    objectives = ("heavy_atoms",)

    def __init__(self, scale: float = 40.0):
        self.scale = scale

    def score(self, molecules: list[Chem.Mol]) -> torch.Tensor:
        return torch.tensor(
            [[mol.GetNumHeavyAtoms() / self.scale] for mol in molecules],
            dtype=torch.float32,
        ).reshape(-1, 1)


def within_heavy_atom_limit(mol: Chem.Mol) -> bool:
    return mol.GetNumHeavyAtoms() <= 40


if __name__ == "__main__":
    config = Config.from_file("configs/qed.yaml")
    reward = HeavyAtomReward(**config.reward.settings)
    trainer = RxnFlowTrainer(
        config,
        reward,
        sample_filter=within_heavy_atom_limit,
    )
    trainer.run()
