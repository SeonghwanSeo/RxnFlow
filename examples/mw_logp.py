"""Minimize molecular weight and maximize logP."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from rxnflow import Config, RxnFlowTrainer
from rxnflow.reward import RewardFunction


class MWReward(RewardFunction):
    name = "mw"

    def __init__(self, scale: float = 300.0):
        self.scale = scale

    def score(self, smiles: str, cond=None) -> float:
        mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
        return -mw / self.scale


class LogPReward(RewardFunction):
    name = "logp"

    def score(self, smiles: str, cond=None) -> float:
        return Descriptors.MolLogP(Chem.MolFromSmiles(smiles))


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
    rewards = [
        MWReward(scale=100.0),
        LogPReward(),
    ]
    trainer = RxnFlowTrainer(
        config,
        rewards,
        seed=args.seed,
        output_dir=args.output_dir,
        device=device,
        num_workers=4,
    )
    checkpoint = trainer.run(
        args.steps, resume_from_checkpoint=args.resume_from_checkpoint
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
