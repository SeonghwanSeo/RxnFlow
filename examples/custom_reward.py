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

    def score(self, mols: list[Chem.Mol]) -> NDArray[np.float32]:
        values = np.zeros((len(mols), 1), dtype=np.float32)
        for i, mol in enumerate(mols):
            num_atoms = mol.GetNumHeavyAtoms()
            # Leave molecules outside the reward's domain at zero.
            if num_atoms <= 40:
                values[i, 0] = num_atoms / self.scale
        return values


if __name__ == "__main__":
    config = Config.from_file("configs/qed.yaml")
    reward = HeavyAtomReward(**config.reward.settings)
    trainer = RxnFlowTrainer(
        config,
        reward,
        output_dir="runs/custom_reward",
    )
    trainer.run(1000)
