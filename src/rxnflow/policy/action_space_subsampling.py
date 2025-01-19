import numpy as np
import torch
from gflownet.utils.misc import get_worker_rng

from rxnflow.config import Config
from rxnflow.envs.env import SynthesisEnv


class ActionSpace:
    def __init__(self, num_actions: int, sampling_ratio: float, min_sampling: int):
        assert sampling_ratio <= 1
        min_sampling = min(num_actions, min_sampling)
        self.num_actions: int = num_actions
        self.num_sampling = max(int(num_actions * sampling_ratio), min_sampling)
        self.sampling_ratio: float = self.num_sampling / self.num_actions

    def sampling(self) -> torch.Tensor:
        # TODO: introduce importance subsampling instead of uniform subsampling
        if self.sampling_ratio < 1:
            rng: np.random.RandomState = get_worker_rng()
            indices = rng.choice(self.num_actions, self.num_sampling, replace=False)
            np.sort(indices)
            return torch.from_numpy(indices).to(torch.long)
        else:
            return torch.arange((self.num_actions), dtype=torch.long)


class SubsamplingPolicy:
    def __init__(self, env: SynthesisEnv, cfg: Config):
        self.global_cfg = cfg
        self.cfg = cfg.algo.action_subsampling

        sr = self.cfg.sampling_ratio
        nmin = int(self.cfg.min_sampling)

        self.block_spaces: dict[str, ActionSpace] = {}
        self.num_blocks: dict[str, int] = {}
        for block_type, blocks in env.blocks.items():
            self.block_spaces[block_type] = ActionSpace(len(blocks), sr, nmin)
        self.sampling_ratios = {t: space.sampling_ratio for t, space in self.block_spaces.items()}

    def sampling(self, block_type: str) -> tuple[torch.Tensor, float]:
        return self.block_spaces[block_type].sampling(), self.sampling_ratios[block_type]
