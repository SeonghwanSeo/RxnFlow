"""Minimal local reward injection example."""

from typing import Any

from rdkit import Chem

from rxnflow import Config, RewardFunction, RxnFlowTrainer


class HeavyAtomReward(RewardFunction):
    name = "heavy_atoms"

    def __init__(self, scale: float = 40.0):
        self.scale = scale

    # Example 1: define score(): smi -> reward
    def score(
        self,
        smiles: str,
        cond: dict[str, Any] | None = None,
    ) -> float:
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumHeavyAtoms() / self.scale if mol is not None else 0.0

    # Example 2: define score_list(): [smi] -> [reward]
    def score_list(
        self,
        smiles_list: list[str],
        cond_list: list[dict[str, Any]] | None = None,
    ) -> list[float]:
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        return [m.GetNumHeavyAtoms() / self.scale if m is not None else 0.0 for m in mols]


if __name__ == "__main__":
    config = Config.from_dict(
        {
            "env_dir": "./data/envs/example_5k/",
            "gflownet": {"beta": 32},
            "property_penalty": {"mw": 500},
        }
    )
    trainer = RxnFlowTrainer(
        config,
        HeavyAtomReward(),
        seed=1,
        output_dir="runs/custom_reward",
        num_workers=4,
    )
    trainer.run(1000)
