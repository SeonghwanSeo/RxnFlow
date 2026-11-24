"""Train a molecular generator conditioned on target TPSA values."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from rxnflow import ConditioningModule, Config, RxnFlowTrainer
from rxnflow.reward import RewardFunction


class CondTPSAReward(RewardFunction):
    """Reward function for TPSA conditioning."""

    name = "tpsa"

    def __init__(self, scale: float = 100.0):
        self.scale = scale

    def score(self, smiles: str, cond: dict[str, Any] | None = None) -> float:
        assert cond is not None, "CondTPSAReward requires conditions"
        mol = Chem.MolFromSmiles(smiles)
        tpsa = Descriptors.TPSA(mol)
        target = cond["tpsa"]
        diff = abs(tpsa - target)
        return -diff / self.scale


class TPSAConditioningModule(ConditioningModule):
    """Conditioning module for TPSA values."""

    encoding_dim = 1

    def __init__(self, min_tpsa: float = 0.0, max_tpsa: float = 200.0):
        self.min_tpsa: float = min_tpsa
        self.max_tpsa: float = max_tpsa

    def sample(
        self,
        num_samples: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        tpsa_values = rng.uniform(self.min_tpsa, self.max_tpsa, size=num_samples)
        return [{"tpsa": float(tpsa)} for tpsa in tpsa_values]

    def encode(self, cond_list: list[dict[str, Any]]) -> np.ndarray:
        tpsa_values = np.array([cond["tpsa"] for cond in cond_list], dtype=np.float32)
        # Scale the TPSA values
        tpsa_values = tpsa_values / 100
        return tpsa_values.reshape(-1, 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tpsa_cond"))
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
    conditioning = TPSAConditioningModule(**config.conditioning)
    reward = CondTPSAReward(scale=100.0)
    trainer = RxnFlowTrainer(
        config,
        reward=reward,
        seed=args.seed,
        output_dir=args.output_dir,
        device=device,
        num_workers=4,
        conditioning=conditioning,
    )
    checkpoint = trainer.run(
        args.steps, resume_from_checkpoint=args.resume_from_checkpoint
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
