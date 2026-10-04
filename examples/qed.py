"""Local QED reward and training example.

Run: python examples/qed.py --config configs/qed.yaml --steps 1000.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED

from rxnflow.config import Config
from rxnflow.reward import RewardFunction
from rxnflow.trainer import RxnFlowTrainer


class QEDReward(RewardFunction):
    objectives = ("qed",)

    def score(self, mols: list[Chem.Mol]) -> np.ndarray:
        qeds = [QED.qed(mol) for mol in mols]
        return np.array(qeds, dtype=np.float32).reshape(-1, 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/qed"))
    parser.add_argument(
        "--steps", type=int, required=True, help="additional training updates"
    )
    parser.add_argument(
        "--device", help="device, e.g. cpu or cuda; omitted selects CUDA when available"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args(argv)
    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = Config.from_file(args.config)
    reward = QEDReward(**config.reward.settings)
    trainer = RxnFlowTrainer(
        config,
        reward,
        output_dir=args.output_dir,
        device=device,
        seed=args.seed,
    )
    checkpoint = trainer.run(
        args.steps, resume_from_checkpoint=args.resume_from_checkpoint
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
