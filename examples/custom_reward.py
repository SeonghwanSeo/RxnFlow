"""Minimal local reward injection example."""

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from rxnflow.config import Config
from rxnflow.reward import RewardFunction
from rxnflow.trainer import RxnFlowTrainer


class HeavyAtomReward(RewardFunction):
    objectives = ("heavy_atoms",)

    def __init__(self, scale: float = 40.0):
        self.scale = scale

    def score(self, molecules: list[Chem.Mol]) -> NDArray[np.float32]:
        return np.array(
            [[mol.GetNumHeavyAtoms() / self.scale] for mol in molecules],
            dtype=np.float32,
        ).reshape(-1, 1)

    def filter_object(self, mol: Chem.Mol) -> bool:
        return mol.GetNumHeavyAtoms() <= 40


if __name__ == "__main__":
    config = Config.from_file("configs/qed.yaml")
    reward = HeavyAtomReward(**config.reward.settings)
    trainer = RxnFlowTrainer(
        config,
        reward,
    )
    trainer.run()
