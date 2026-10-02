"""QED/SA with beta and preference conditioning: python examples/qed_sa.py --config configs/qed_sa.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer

from rxnflow.config import Config
from rxnflow.reward import RewardFunction
from rxnflow.trainer import RxnFlowTrainer


class QEDSAReward(RewardFunction):
    """Maximize QED and synthetic accessibility; objective order is QED, SA."""

    objectives = ("qed", "sa")

    def score(self, molecules: list[Chem.Mol]) -> NDArray[np.float32]:
        # SA score ranges from 1 (easy) to 10 (difficult); reward is larger-is-better.
        return np.array(
            [
                [QED.qed(mol), (10.0 - sascorer.calculateScore(mol)) / 9.0]
                for mol in molecules
            ],
            dtype=np.float32,
        ).reshape(-1, 2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args(argv)
    config = Config.from_file(args.config)
    reward = QEDSAReward(**config.reward.settings)
    trainer = RxnFlowTrainer(config, reward, restart=args.restart)
    checkpoint = trainer.run(args.steps)
    print(checkpoint)


if __name__ == "__main__":
    main()
