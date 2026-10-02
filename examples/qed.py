"""Local QED reward and training example: python -m examples.qed --config configs/qed.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import QED

from rxnflow.config import Config
from rxnflow.reward import RewardFunction
from rxnflow.trainer import RxnFlowTrainer


class QEDReward(RewardFunction):
    """Example reward using RDKit's quantitative estimate of drug-likeness."""

    objectives = ("qed",)

    def score(self, molecules: list[Chem.Mol]) -> NDArray[np.float32]:
        return np.array([QED.qed(mol) for mol in molecules], dtype=np.float32).reshape(
            -1, 1
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args(argv)
    config = Config.from_file(args.config)
    reward = QEDReward(**config.reward.settings)
    trainer = RxnFlowTrainer(config, reward, restart=args.restart)
    try:
        checkpoint = trainer.run(args.steps)
        print(checkpoint)
    finally:
        if "retro_analyzer" in trainer.env.__dict__:
            trainer.env.retro_analyzer.close()


if __name__ == "__main__":
    main()
