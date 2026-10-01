"""Minimal local reward injection example."""

from rxnflow.config import Config
from rxnflow.gflownet.types import Sample
from rxnflow.reward import RewardFunction
from rxnflow.trainer import RxnFlowTrainer


class HeavyAtomReward(RewardFunction):
    def __init__(self, scale: float = 40.0):
        self.scale = scale

    def score(self, samples: list[Sample]) -> list[float]:
        return [sample.mol.GetNumHeavyAtoms() / self.scale for sample in samples]


def within_heavy_atom_limit(sample: Sample) -> bool:
    return sample.mol.GetNumHeavyAtoms() <= 40


if __name__ == "__main__":
    config = Config.from_file("configs/qed.yaml")
    reward = HeavyAtomReward(**config.reward.settings)
    trainer = RxnFlowTrainer(
        config,
        reward,
        sample_filter=within_heavy_atom_limit,
    )
    trainer.run()
