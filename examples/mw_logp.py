"""Minimize molecular weight and maximize logP."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from rxnflow.config import Config
from rxnflow.reward import RewardFunction
from rxnflow.trainer import RxnFlowTrainer


class MWLogPReward(RewardFunction):
    objectives = ("mw", "logp")

    def __init__(self, mw_scale: float = 300.0, logp_scale: float = 1.0):
        self.mw_scale = mw_scale
        self.logp_scale = logp_scale

    def score(self, mols: list[Chem.Mol]) -> np.ndarray:
        # Convert both properties to positive rewards: lower MW and higher logP.
        rewards = [
            [
                1 / (1 + Descriptors.ExactMolWt(mol) / self.mw_scale),
                1 / (1 + np.exp(-Descriptors.MolLogP(mol) / self.logp_scale)),
            ]
            for mol in mols
        ]
        return np.array(rewards, dtype=np.float32).reshape(-1, 2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mw_logp"))
    parser.add_argument(
        "--steps", type=int, required=True, help="additional training updates"
    )
    parser.add_argument(
        "--device", help="device, e.g. cpu or cuda; omitted selects CUDA when available"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args(argv)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config.from_file(args.config)
    reward = MWLogPReward(**config.reward.settings)
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
