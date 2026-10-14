"""Train a molecular generator conditioned on target TPSA values."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from rxnflow import ConditioningModule, Config, RewardFunction, RxnFlowTrainer


class TPSAReward(RewardFunction):
    """Reward function for TPSA conditioning."""

    objectives = ("tpsa",)

    def score(
        self,
        smiles_list: list[str],
        cond_list: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        assert cond_list is not None, "TPSAReward requires conditions"

        # Compute TPSA for each molecule
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        sample_tpsa = np.array([Descriptors.TPSA(mol) for mol in mols], dtype=np.float32)

        # Compute the difference between sample TPSA and target TPSA from conditions
        target_tpsa = np.array([cond["tpsa"] for cond in cond_list], dtype=np.float32)
        diff = np.abs(sample_tpsa - target_tpsa)

        # Compute rewards
        rewards = 1 / (1 + diff)
        return rewards.reshape(-1, 1)


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
    conditioning_module = TPSAConditioningModule(**config.conditioning)
    reward = TPSAReward(**config.reward.settings)
    trainer = RxnFlowTrainer(
        config,
        reward,
        seed=args.seed,
        output_dir=args.output_dir,
        device=device,
        num_workers=4,
        conditioning=conditioning_module,
    )
    checkpoint = trainer.run(
        args.steps, resume_from_checkpoint=args.resume_from_checkpoint
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
